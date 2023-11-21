import json
import spacy
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import torch
import gensim.downloader as api
import numpy as np

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
word2vec_model = api.load("word2vec-google-news-300") # carregar modelo Word2Vec pré-treinado

# ---------------------------------------------------------------------------
# Função para gerar Word2Vec Embeddings (2º representação)
def get_word2vec_embeddings(model, text):

    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    # Média dos embeddings para representar todo o texto
    return np.mean(embeddings, axis=0) if len(embeddings) > 0 else np.zeros(300)

# Função de preparação das representações -> TF-IDF e Word2Vec
def prepare_data(conv, quest, options):

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode')
    tfidf_vectorizer.fit([conv] + [quest] + options)
    conv_tfidf = tfidf_vectorizer.transform([conv])
    quest_tfidf = tfidf_vectorizer.transform([quest])
    options_tfidf = tfidf_vectorizer.transform(options)

    # Word2Vec
    conv_w2v = get_word2vec_embeddings(word2vec_model, conv)
    quest_w2v = get_word2vec_embeddings(word2vec_model, quest)
    options_w2v = np.array([get_word2vec_embeddings(word2vec_model, opt) for opt in options])

    return conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v
# ---------------------------------------------------------------------------

# Pré-processamento dos dados -> lematização, remover potuação, etc.
def preprocessing(conv, quest, options):

    # Escolher apenas as falas do diálogo correspondentes ao sujeito principal da pergunta
    dep_question = {}
    for token in nlp(quest):
        dep_question[token.text] = token.dep_

    man_in_question = re.findall("\\bthe man\\b", quest)
    woman_in_question = re.findall("\\bthe woman\\b", quest)

    if (len(woman_in_question) > 0 and len(man_in_question) == 0 and dep_question["woman"] == "nsubj"):

        dialogue = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))
                    if conv[i].startswith("W:") or conv[i].startswith("F:") or conv[i].startswith("Woman:")]

    elif (len(man_in_question) > 0 and len(woman_in_question) == 0 and dep_question["man"] == "nsubj"):

        dialogue = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))
                    if conv[i].startswith("M:") or conv[i].startswith("Man:")]

    else:
        dialogue = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))]

    if dialogue == []:
        dialogue = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))]

    # Lematização
    conv_lemmas = []
    for sentence in dialogue:
        sentence_lemmas = []
        for conv_token in nlp(sentence):
            sentence_lemmas.append(conv_token.lemma_)
        conv_lemmas.append(sentence_lemmas)
    
    quest_lemmas = []
    for quest_token in nlp(quest):
        quest_lemmas.append(quest_token.lemma_)

    opt_lemmas = []
    for opt in options:
        option_lemmas = []
        for opt_token in nlp(opt):
            option_lemmas.append(opt_token.lemma_)
        opt_lemmas.append(option_lemmas)

    # Remover pontuação
    conv = [remove_punct(sent) for sent in conv_lemmas]
    quest = " ".join(remove_punct(quest_lemmas))
    options = [" ".join(remove_punct(opt)) for opt in opt_lemmas]

    # Remover stopwords
    conv = [" ".join(remove_stop(sent)) for sent in conv]

    return conv, quest, options

# Remover pontuação de uma lista de palavras
def remove_punct(text):
    import string
    punct = string.punctuation

    new_text = []
    for word in text:
        if word not in punct:
            new_text.append(word)

    return new_text

# Remover stopwords de uma lista de palavras
def remove_stop(text):
    stopwords = nltk.corpus.stopwords.words('english')

    new_text = []
    for word in text:
        if word not in stopwords:
            new_text.append(word)

    return new_text

# ---------------------------------------------------------------------------

# Topic Modelling
def topic_modelling(conv, quest, options, options_w2v):
    from sklearn.decomposition import LatentDirichletAllocation

    # TF-IDF
    tf_vect = TfidfVectorizer(ngram_range=(1, 3), strip_accents='unicode')
    tf_vect.fit([conv])

    sentences = nltk.sent_tokenize(conv)
    conv_vect = [tf_vect.transform([sentence]) for sentence in sentences]
    quest_vect = tf_vect.transform([question])
    opt_vect = [tf_vect.transform([options[i]]) for i in range(len(options))]

    lda = LatentDirichletAllocation(n_components=2, doc_topic_prior=0.5, topic_word_prior=0.5)

    best = []
    for i in range(len(sentences)):
        lda.fit(conv_vect[i])
        quest_topics = lda.transform(quest_vect)[0]
        best.append(np.sum(quest_topics))

    best_sentence = conv_vect[np.argmax(best)]

    lda.fit_transform(best_sentence)[0]
    opt_topics = [lda.transform(opt_vect[i])[0] for i in range(len(opt_vect))]

    similarity = [np.sum(opt_topics[i]) + 0.4*cosine_similarity(best_sentence, opt_vect[i])
                  for i in range(len(opt_vect))]
    correct_option_tfidf = options[np.argmax(similarity)]

    # Usar similaridade de cosseno para Word2Vec (O LDA não é aplicado diretamente a embeddings densos como Word2Vec)
    quest_w2v = get_word2vec_embeddings(word2vec_model, quest)
    sent_w2v = np.array([get_word2vec_embeddings(word2vec_model, sentence) for sentence in sentences])

    best = []
    for i in range(len(sentences)):
        sim = cosine_similarity([sent_w2v[i]], [quest_w2v])[0][0]
        best.append(sim)

    best_sentence = sent_w2v[np.argmax(best)]

    similarity = [cosine_similarity([best_sentence], [opt_emb])[0][0] for opt_emb in options_w2v]
    correct_option_w2v = options[np.argmax(similarity)]

    return correct_option_tfidf, correct_option_w2v


# Support Vector Machines - classificação binária
def svm_prediction(conv, options):
    from sklearn.svm import SVC

    conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v = prepare_data(conv, options)

    # SVM com TF-IDF
    svm_model_tfidf = SVC(kernel='linear')
    svm_model_tfidf.fit(conv_tfidf, [0])
    predictions_tfidf = svm_model_tfidf.predict(options_tfidf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf)]

    # SVM com Word2Vec
    svm_model_w2v = SVC(kernel='linear')
    svm_model_w2v.fit([conv_w2v], [0])
    predictions_w2v = svm_model_w2v.predict(options_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v)]

    return pred_option_tfidf, pred_option_w2v


# Árvore de Decisão
def tree_prediction_both(conv, options):
    from sklearn.tree import DecisionTreeClassifier

    conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v = prepare_data(conv, options)

    # Árvore de Decisão com TF-IDF
    tree_model_tfidf = DecisionTreeClassifier()
    tree_model_tfidf.fit(conv_tfidf, [0])
    predictions_tfidf = tree_model_tfidf.predict(options_tfidf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf)]

    # Árvore de Decisão com Word2Vec
    tree_model_w2v = DecisionTreeClassifier()
    tree_model_w2v.fit([conv_w2v], [0])
    predictions_w2v = tree_model_w2v.predict(options_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v)]

    return pred_option_tfidf, pred_option_w2v

def naive_bayes(conv, options):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

    conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v = prepare_data(conv, options)

    # Naive Bayes com TF-IDF
    model.fit(conv_tfidf, [0])
    predictions_tfidf = model.predict(options_tfidf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf)]

    # Naive Bayes com Word2Vec
    model.fit([conv_w2v], [0])
    predictions_w2v = model.predict(options_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v)]

    return pred_option_tfidf, pred_option_w2v

# ---------------------------------------------------------------------------

# Métricas de avaliação
def evaluation_metrics(true_answer, predicted_answer):
    print("Nº total de questões:", len(true_answer))

    # Exact Match
    exact_match = 0
    for (true, predicted) in zip(true_answer, predicted_answer):
        if true == predicted:
            exact_match += 1
    print("EXACT MATCH:", exact_match)

    # F1-score
    f1 = f1_score(true_answer, predicted_answer, average='weighted')
    print("F1-SCORE:", f1)

    # Similaridade - uso do coseno
    tf_vect = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode', max_features=500, min_df=3, max_df=0.5)

    true = tf_vect.fit_transform(true_answer)
    predicted = tf_vect.transform(predicted_answer)

    cosine_matrix = cosine_similarity(true, predicted)
    similarity = sum(cosine_matrix[i][i] for i in range(len(cosine_matrix)))
    print("SIMILARIDADE:", similarity)

    # Chamar a função do SAS
    sas_score = calculate_sas(true_answer, predicted_answer)
    print("SAS:", sas_score)

# Calcular SAS
def calculate_sas(true_answer, predicted_answer, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):

    # Carregar modelo e tokenizer pré-treinados
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Obter embeddings
    def get_embeddings(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state
        return torch.mean(embeddings, dim=1)

    # Embeddings para respostas reais e previstas
    true_embeddings = get_embeddings(true_answer)
    predicted_embeddings = get_embeddings(predicted_answer)

    # Calcular SAS
    sas = torch.nn.functional.cosine_similarity(true_embeddings, predicted_embeddings, dim=1)
    sas_score = torch.mean(sas).item()

    return sas_score

# ---------------------------------------------------------------------------
with open('test.json', 'r') as file:
    data = json.load(file)

true_answers = []
predicted_answers_topic_tfidf = []
predicted_answers_topic_w2v = []
predicted_answers_svm_tfidf = []
predicted_answers_svm_w2v = []
predicted_answers_tree_tfidf = []
predicted_answers_tree_w2v = []
predicted_answers_nb_tfidf = []
predicted_answers_nb_w2v = []

for conversation_data in data:
    for qa_pair in conversation_data[1]:
        
        conversation_text = conversation_data[0]

        question = qa_pair['question'].lower()
        choices = [qa_pair['choice'][i].lower() for i in range(len(qa_pair['choice']))]
        answer = qa_pair['answer'].lower()

        # Realizar pré-processamento nos dados
        conversation_text, question, choices = preprocessing(conversation_text, question, choices)

        print(conversation_text, "\n", question, "\n", choices)

        # Preparar os dados para as representações
        conv_tfidf, options_tfidf, conv_w2v, options_w2v = prepare_data(conversation_text, choices)

        # Chamar as funções de previsão para cada modelo
        pred_svm_tfidf, pred_svm_w2v = svm_prediction(conversation_text, choices)
        pred_tree_tfidf, pred_tree_w2v = tree_prediction_both(conversation_text, choices)
        pred_topic_tfidf, pred_topic_w2v = topic_modelling(conversation_text, question, choices, options_w2v)
        pred_nb_tfidf, pred_nb_w2v = naive_bayes(conversation_text, choices)

        # Armazenar as respostas verdadeiras e as previstas
        true_answers.append(answer)
        predicted_answers_svm_tfidf.append(pred_svm_tfidf)
        predicted_answers_svm_w2v.append(pred_svm_w2v)
        predicted_answers_tree_tfidf.append(pred_tree_tfidf)
        predicted_answers_tree_w2v.append(pred_tree_w2v)
        predicted_answers_topic_tfidf.append(pred_topic_tfidf)
        predicted_answers_topic_w2v.append(pred_topic_w2v)
        predicted_answers_nb_tfidf.append(pred_nb_tfidf)
        predicted_answers_nb_w2v.append(pred_nb_w2v)

    break

print('SVM: TF-IDF ----------------------------')
evaluation_metrics(true_answers, predicted_answers_svm_tfidf)
print('SVM: Word2Vec --------------------------')
evaluation_metrics(true_answers, predicted_answers_svm_w2v)
print('Árvore de decisão: TF-IDF --------------')
evaluation_metrics(true_answers, predicted_answers_tree_tfidf)
print('Árvore de decisão: Word2Vec ------------')
evaluation_metrics(true_answers, predicted_answers_tree_w2v)
print('Naive Bayes: TF-IDF --------------------')
evaluation_metrics(true_answers, predicted_answers_nb_tfidf)
print('Naive Bayes: Word2Vec ------------------')
evaluation_metrics(true_answers, predicted_answers_nb_w2v)
print('Topic Modelling: TF-IDF -----------------')
evaluation_metrics(true_answers, predicted_answers_topic_tfidf)
print('Topic Modelling: Word2Vec ---------------')
evaluation_metrics(true_answers, predicted_answers_topic_w2v)