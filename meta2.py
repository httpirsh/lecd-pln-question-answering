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
import scipy

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
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')
    tfidf_vectorizer.fit(conv + [quest] + options)
    conv_tfidf = [tfidf_vectorizer.transform([sentence]) for sentence in conv]
    quest_tfidf = tfidf_vectorizer.transform([quest])
    options_tfidf = [tfidf_vectorizer.transform([option]) for option in options]

    # Word2Vec
    conv_w2v = np.array([get_word2vec_embeddings(word2vec_model, sentence) for sentence in conv])
    quest_w2v = get_word2vec_embeddings(word2vec_model, quest)
    options_w2v = np.array([get_word2vec_embeddings(word2vec_model, opt) for opt in options])

    return conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v

# ---------------------------------------------------------------------------

# Pré-processamento dos dados -> lematização, remover potuação, etc.
# Aplicar à resposta correta para conseguir comparar com a resposta prevista pelos vários métodos
def preprocessing(conv, quest, options, answer):

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

    ans_lemmas = []
    for ans_token in nlp(answer):
        ans_lemmas.append(ans_token.lemma_)

    # Remover pontuação
    conv = [remove_punct(sent) for sent in conv_lemmas]
    quest = " ".join(remove_punct(quest_lemmas))
    options = [" ".join(remove_punct(opt)) for opt in opt_lemmas]
    ans = " ".join(remove_punct(ans_lemmas))

    # Remover stopwords
    conv = [" ".join(remove_stop(sent)) for sent in conv]

    return conv, quest, options, ans

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

# Dados de treino para os modelos
def train_data():
    X_tfidf = []
    X_w2v = []
    y = []

    all_text = []

    text = []
    questions = []
    opt = []
    ans = []

    with open('train.json', 'r') as file:
        data = json.load(file)
    
    n = 1
    for conversation_data in data:
        for qa_pair in conversation_data[1]:
            
            conversation_text = conversation_data[0]

            question = qa_pair['question'].lower()
            choices = [qa_pair['choice'][i].lower() for i in range(len(qa_pair['choice']))]
            answer = qa_pair['answer'].lower()

            # Realizar pré-processamento nos dados
            conversation_text, question, options, answer = preprocessing(conversation_text, question, choices, answer)
            text.append(conversation_text)
            questions.append(question)
            opt.append(options)
            ans.append(answer)

            all_text.extend(conversation_text)
            all_text.append(question)
            all_text.extend(options)

        n += 1
        if n > 500: # Usar 500 perguntas como dados de treino
            break

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')
    tfidf_vectorizer.fit(all_text)

    n=1
    for c, q, o, a in zip(text, questions, opt, ans):

        for choice in o:
            c.append(q)
            c.append(choice)
            all = " ".join(c)
            all_tfidf = tfidf_vectorizer.transform([all])
            all_w2v = get_word2vec_embeddings(word2vec_model, all)

            X_tfidf.append(all_tfidf)
            X_w2v.append(all_w2v)
            y.append(1 if choice == a else 0)

    X_tfidf = scipy.sparse.vstack(X_tfidf)
    return X_tfidf.toarray(), np.array(X_w2v), np.array(y), tfidf_vectorizer

# ---------------------------------------------------------------------------

# Topic Modelling
def topic_modelling(conv, quest, options):
    from sklearn.decomposition import LatentDirichletAllocation

    conv_tfidf, quest_tfidf, options_tfidf, conv_w2v, quest_w2v, options_w2v = prepare_data(conv, quest, options)

    # TF-IDF
    lda = LatentDirichletAllocation(n_components=2, doc_topic_prior=0.5, topic_word_prior=0.5)

    best = []
    for i in range(len(conv_tfidf)):
        lda.fit(conv_tfidf[i])
        quest_topics = lda.transform(quest_tfidf)[0]
        best.append(np.sum(quest_topics))

    best_sentence = conv_tfidf[np.argmax(best)]

    lda.fit_transform(best_sentence)[0]
    opt_topics = [lda.transform(options_tfidf[i])[0] for i in range(len(options_tfidf))]

    similarity = [np.sum(opt_topics[i]) + 0.4*cosine_similarity(best_sentence, options_tfidf[i])
                  for i in range(len(options_tfidf))]
    correct_option_tfidf = options[np.argmax(similarity)]

    # Usar similaridade de cosseno para Word2Vec (O LDA não é aplicado diretamente a embeddings densos como Word2Vec)
    best = []
    for i in range(len(conv_w2v)):
        sim = cosine_similarity([conv_w2v[i]], [quest_w2v])[0][0]
        best.append(sim)

    best_sentence = conv_w2v[np.argmax(best)]

    similarity = [cosine_similarity([best_sentence], [opt])[0][0] for opt in options_w2v]
    correct_option_w2v = options[np.argmax(similarity)]

    return correct_option_tfidf, correct_option_w2v


# Support Vector Machines - classificação binária
def svm_prediction(model_tf, model_w2v, x_tf, x_w2v, options):

    # SVM com TF-IDF
    predictions_tfidf = model_tf.predict_proba(x_tf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf[:,1])]

    # SVM com Word2Vec
    predictions_w2v = model_w2v.predict_proba(x_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v[:,1])]

    return pred_option_tfidf, pred_option_w2v


# Árvore de Decisão
def tree_prediction_both(model_tf, model_w2v, x_tf, x_w2v, options):

    # Árvore de Decisão com TF-IDF
    predictions_tfidf = model_tf.predict_proba(x_tf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf[:,1])]

    # Árvore de Decisão com Word2Vec
    predictions_w2v = model_w2v.predict_proba(x_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v[:,1])]

    return pred_option_tfidf, pred_option_w2v


# Naive Bayes
def naive_bayes(model_tf, model_w2v, x_tf, x_w2v, options):

    # Naive Bayes com TF-IDF
    predictions_tfidf = model_tf.predict_proba(x_tf)
    pred_option_tfidf = options[np.argmax(predictions_tfidf[:,1])]

    # Naive Bayes com Word2Vec
    predictions_w2v = model_w2v.predict_proba(x_w2v)
    pred_option_w2v = options[np.argmax(predictions_w2v[:,1])]

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

X_tfidf, X_w2v, y, tf_vect = train_data()

# Treino dos modelos
from sklearn.svm import SVC
svm_model_tfidf = SVC(kernel='linear', probability=True)
svm_model_tfidf.fit(X_tfidf, y)
svm_model_w2v = SVC(kernel='linear', probability=True)
svm_model_w2v.fit(X_w2v, y)

from sklearn.tree import DecisionTreeClassifier
tree_model_tfidf = DecisionTreeClassifier()
tree_model_tfidf.fit(X_tfidf, y)
tree_model_w2v = DecisionTreeClassifier()
tree_model_w2v.fit(X_w2v, y)

from sklearn.naive_bayes import GaussianNB
model_tfidf = GaussianNB()
model_tfidf.fit(X_tfidf, y)
model_w2v = GaussianNB()
model_w2v.fit(X_w2v, y)

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
        conversation_text, question, choices, answer = preprocessing(conversation_text, question, choices, answer)

        # Organizar dados para serem classificados pelos modelos
        Xt_tfidf, Xt_w2v, yt = [], [], []
        for choice in choices:

            conversation_text.append(question)
            conversation_text.append(choice)
            all = " ".join(conversation_text)

            all_tfidf = tf_vect.transform([all])
            all_w2v = get_word2vec_embeddings(word2vec_model, all)

            Xt_tfidf.append(all_tfidf)
            Xt_w2v.append(all_w2v)
            yt.append(1 if choice == answer else 0)

        Xt_tfidf = scipy.sparse.vstack(Xt_tfidf)
        Xt_tfidf = Xt_tfidf.toarray()

        # Chamar as funções de previsão para cada modelo
        pred_svm_tfidf, pred_svm_w2v = svm_prediction(svm_model_tfidf, svm_model_w2v, Xt_tfidf, Xt_w2v, choices)
        pred_tree_tfidf, pred_tree_w2v = tree_prediction_both(tree_model_tfidf, tree_model_w2v, Xt_tfidf, Xt_w2v, choices)
        pred_topic_tfidf, pred_topic_w2v = topic_modelling(conversation_text, question, choices)
        pred_nb_tfidf, pred_nb_w2v = naive_bayes(model_tfidf, model_w2v, Xt_tfidf, Xt_w2v, choices)

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