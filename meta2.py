import json
import spacy
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import numpy as np

nlp = spacy.load("en_core_web_sm")
word2vec_model = api.load("word2vec-google-news-300") # carregar modelo Word2Vec pré-treinado

# -------------------------------------------------------------------------
# Função para gerar Word2Vec Embeddings (2º representação)
def get_word2vec_embeddings(model, text):

    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    # Média dos embeddings para representar todo o texto
    return np.mean(embeddings, axis=0) if len(embeddings) > 0 else np.zeros(300)

# Função de preparação das representações -> TF-IDF e Word2Vec
def prepare_data(conv, options):

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode')
    tfidf_vectorizer.fit([conv] + options)
    conv_tfidf = tfidf_vectorizer.transform([conv])
    options_tfidf = tfidf_vectorizer.transform(options)

    # Word2Vec
    conv_w2v = get_word2vec_embeddings(word2vec_model, conv)
    options_w2v = np.array([get_word2vec_embeddings(word2vec_model, opt) for opt in options])

    return conv_tfidf, options_tfidf, conv_w2v, options_w2v
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

        conv = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))
                if conv[i].startswith("W:") or conv[i].startswith("F:") or conv[i].startswith("Woman:")]

    elif (len(man_in_question) > 0 and len(woman_in_question) == 0 and dep_question["man"] == "nsubj"):

        conv = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))
                if conv[i].startswith("M:") or conv[i].startswith("Man:")]

    else:
        conv = [conv[i][conv[i].index(":")+2:].lower() for i in range(len(conv))]
        
    

# ---------------------------------------------------------------------------

# Topic Modelling
def topic_modelling(conv, quest, options, conv_w2v, options_w2v):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity

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

    sent_topics = lda.fit_transform(best_sentence)[0]
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

    conv_tfidf, options_tfidf, conv_w2v, options_w2v = prepare_data(conv, options)

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

    conv_tfidf, options_tfidf, conv_w2v, options_w2v = prepare_data(conv, options)

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


# -------------------------------------------------------------------
with open('test.json', 'r') as file:
    data = json.load(file)

true_answers = []
predicted_answers_topic_tfidf = []
predicted_answers_topic_w2v = []
predicted_answers_svm_tfidf = []
predicted_answers_svm_w2v = []
predicted_answers_tree_tfidf = []
predicted_answers_tree_w2v = []

for conversation_data in data:
    for qa_pair in conversation_data[1]:
        
        conversation_text = conversation_data[0]

        question = qa_pair['question'].lower()
        choices = [qa_pair['choice'][i].lower() for i in range(len(qa_pair['choice']))]
        answer = qa_pair['answer'].lower()

        nlp_question = nlp(question)
        nlp_choices = [nlp(choice) for choice in choices]
        nlp_answer = nlp(answer)

        dep_question = {}
        for token in nlp_question:
            dep_question[token.text] = token.dep_

        man_in_question = re.findall("\\bthe man\\b", question)
        woman_in_question = re.findall("\\bthe woman\\b", question)

        if (len(woman_in_question) > 0 and len(man_in_question) == 0
                and dep_question["woman"] == "nsubj"):
            conversation_text = [conversation_text[i][conversation_text[i].index(":")+2:] for i in
                                 range(len(conversation_text))
                                 if conversation_text[i].startswith("W:") or conversation_text[i].startswith("F:")
                                 or conversation_text[i].startswith("Woman:")]

        elif (len(man_in_question) > 0 and len(woman_in_question) == 0
              and dep_question["man"] == "nsubj"):
            conversation_text = [conversation_text[i][conversation_text[i].index(":")+2:] for i in
                                 range(len(conversation_text))
                                 if conversation_text[i].startswith("M:") or conversation_text[i].startswith("Man:")]

        else:
            conversation_text = [conversation_text[i][conversation_text[i].index(":")+2:] for i in
                                 range(len(conversation_text))]

        if conversation_text == []:
            break

        conversation_text = " ".join(conversation_text)
        nlp_conv = nlp(conversation_text)

        # Preparar os dados para as representações
        conv_tfidf, options_tfidf, conv_w2v, options_w2v = prepare_data(conversation_text, choices)

        # Chamar as funções de previsão para cada modelo
        #pred_svm_tfidf, pred_svm_w2v = svm_prediction(conversation_text, choices)
        #pred_tree_tfidf, pred_tree_w2v = tree_prediction_both(conversation_text, choices)
        pred_topic_tfidf, pred_topic_w2v = topic_modelling(conversation_text, question, choices, conv_w2v, options_w2v)

        # Armazenar as respostas verdadeiras e as previstas
        true_answers.append(answer)
        predicted_answers_svm_tfidf.append(pred_svm_tfidf)
        predicted_answers_svm_w2v.append(pred_svm_w2v)
        predicted_answers_tree_tfidf.append(pred_tree_tfidf)
        predicted_answers_tree_w2v.append(pred_tree_w2v)
        predicted_answers_topic_tfidf.append(pred_topic_tfidf)
        predicted_answers_topic_w2v.append(pred_topic_w2v)