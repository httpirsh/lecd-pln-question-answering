import json
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import string
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

with open('train.json', 'r') as file:
    data = json.load(file)

nlp = spacy.load("en_core_web_sm")
punct = string.punctuation

# Extrair n-gramas e entidades
def extract_bigrams(text):
    tokens = text.split()

    bigrams = []
    for i in range(len(tokens)-1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        bigrams.append(bigram)
    return bigrams

'''
Bigramas extraídos:
Conversa: ['M: How', 'How long', ..., 'something new.']
Pergunta: ["What's the", 'the woman', 'woman probably', 'probably going', 'going to', 'to do?']
Opções de resposta: [['To teach', 'teach a', 'a different', 'different textbook.'], ['To change', 'change her', 'her job.'], ['To learn', 'learn a', 'a different', 'different textbook.']]
'''

# Análise de sentimentos
def analyze_sentiment(text):
    positive_words = ["love", "enjoy", "good", "happy", "like", "wonderful", "great"]
    negative_words = ["hate", "dislike", "bad", "sad", "terrible", "horrible", "tired"]
    
    positive_count = sum(word in text.lower().split() for word in positive_words)
    negative_count = sum(word in text.lower().split() for word in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Prever a resposta
def predict_answer(conversation, question, choices):

    scores = [0] * len(choices)
    
    # Regra 1: Bigramas
    conversation_bigrams = set(extract_bigrams(conversation))
    question_bigrams = set(extract_bigrams(question))
    choices_bigrams = [set(extract_bigrams(choice)) for choice in choices]
    for i, choice_bigrams in enumerate(choices_bigrams):
        scores[i] += len(conversation_bigrams.intersection(choice_bigrams).intersection(question_bigrams))
    
    # Regra 2: Entidades
    '''
    conversation_entities = set(extract_entities(conversation))
    question_entities = set(extract_entities(question))
    choices_entities = [set(extract_entities(choice)) for choice in choices]
    for i, choice_entities in enumerate(choices_entities):
        scores[i] += len(conversation_entities.intersection(choice_entities).intersection(question_entities))'''

    # Regra 3: Sentimento
    conversation_sentiment = analyze_sentiment(conversation)

    for i, choice in enumerate(choices):
        choice_sentiment = analyze_sentiment(choice)
        # Adicionar score se o sentimento da opção de resposta corresponder ao sentimento
        scores[i] += (conversation_sentiment == choice_sentiment)
        # Adicionar score se o sentimento da opção de resposta corresponder ao sentimento da pergunta
        question_sentiment = analyze_sentiment(question)
        scores[i] += (question_sentiment == choice_sentiment)
 
    # Regra 4: Funções Gramaticais e Relações
    verbs = []
    names = []

    sentences = nltk.sent_tokenize(conversation)
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            if tag[1].startswith("V") and tag[0] not in verbs:
                v = nlp(tag[0])
                verbs.append(v[0].lemma_)
            if tag[1].startswith("N") and tag[0] not in names:
                n = nlp(tag[0])
                names.append(n[0].lemma_)

    for i, choice in enumerate(choices):
        lemmas = [token.lemma_ for token in nlp(choice)]

        scores[i] += sum(verbo in lemmas for verbo in verbs)
        scores[i] += sum(nome in lemmas for nome in names)

    # Prever a opção de resposta com a pontuação mais alta
    predicted_idx = scores.index(max(scores))
    return choices[predicted_idx]

# Funções necessárias para o SAS
def calculate_sas(true_answer, predicted_answer, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):

    # Carregar modelo e tokenizer pré-treinados
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Funciton to get embeddings
    def get_embeddings(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state
        # We use the mean of the token embeddings for the sentence embeddings
        return torch.mean(embeddings, dim=1)

    # Get embeddings for true and predicted answers
    true_embeddings = get_embeddings(true_answer)
    predicted_embeddings = get_embeddings(predicted_answer)

    # Calculate SAS
    sas = torch.nn.functional.cosine_similarity(true_embeddings, predicted_embeddings, dim=1)
    sas_score = torch.mean(sas).item()  # get the mean as a Python scalar

    return sas_score

# Avaliações Métricas
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

def remove_punct(text):
    text = text.split(" ")
    new_text = []
    for word in text:
        for car in word:
            if car in punct:
                word = word.replace(car, "")
        new_text.append(word)

    return " ".join(new_text)

def results(data):

    true_answers = []
    predicted_answers = []

    for conversation_data in data:
        # PRÉ-PROCESSAMENTO

        for qa_pair in conversation_data[1]:
            conversation_text = conversation_data[0]

            question = remove_punct(qa_pair['question'].lower())
            choices = qa_pair['choice']
            answer = remove_punct(qa_pair['answer'].lower())

            for i in range(len(choices)):
                choices[i] = remove_punct(choices[i].lower())

            quest_nlp = nlp(question)
            tokens_question = {}

            for token in quest_nlp:
                tokens_question[token.text] = token.dep_

            if ("\\bthe woman\\b" in question and "\\bthe man\\b" not in question
                    and tokens_question["woman"] == "nsubj"):
                conversation_text = [conversation_text[i][conversation_text[i].index(":"):] for i in range(len(conversation_text))
                                     if conversation_text[i].startswith("W:") or conversation_text[i].startswith("F:")]

            elif ("\\bthe man\\b" in question and "\\bthe woman\\b" not in question
                  and tokens_question["man"] == "nsubj"):
                conversation_text = [conversation_text[i][conversation_text[i].index(":"):] for i in range(len(conversation_text))
                                     if conversation_text[i].startswith("M:")]

            else:
                conversation_text = [conversation_text[i][conversation_text[i].index(":"):] for i in range(len(conversation_text))]

            conversation_text = " ".join(conversation_text)

            predicted_answer = predict_answer(conversation_text, question, choices)

            true_answers.append(answer)
            predicted_answers.append(predicted_answer)

    evaluation_metrics(true_answers, predicted_answers)

results(data)
