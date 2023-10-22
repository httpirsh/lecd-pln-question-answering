import json
import spacy
import re
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

with open('dev.json', 'r') as file:
    data = json.load(file)

nlp = spacy.load("en_core_web_sm")

conversation = " ".join(data[0][0])
question = data[0][1][0]['question']
choices = data[0][1][0]['choice']
answer = data[0][1][0]['answer']

# Processar tudo com o spaCy
conversation_nlp = nlp(conversation)
question_nlp = nlp(question)
choices_nlp = [nlp(choice) for choice in choices]

# Extrair n-gramas e entidades
def extract_bigrams(text):
    tokens = text.split()

    bigrams = []
    for i in range(len(tokens)-1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        bigrams.append(bigram)
    return bigrams

def extract_entities(text):
    return re.findall(r'\b[A-Z][a-z]*\b', text)

'''
Bigramas extraídos:
Conversa: ['M: How', 'How long', ..., 'something new.']
Pergunta: ["What's the", 'the woman', 'woman probably', 'probably going', 'going to', 'to do?']
Opções de resposta: [['To teach', 'teach a', 'a different', 'different textbook.'], ['To change', 'change her', 'her job.'], ['To learn', 'learn a', 'a different', 'different textbook.']]

Entidades Potenciais (palavras com a primeira letra maiúscula):
Conversa: ['M', 'How', 'W', 'For', 'To', 'I', 'I', 'I']
Pergunta: ['What']
Opções de resposta: [['To'], ['To'], ['To']]
'''

# Análise de sentimentos
positive_words = ["love", "enjoy", "good", "happy", "like", "wonderful", "great"]
negative_words = ["hate", "dislike", "bad", "sad", "terrible", "horrible"]

def analyze_sentiment(text):
    
    positive_count = sum(word.lower() in text.lower().split() for word in positive_words)
    negative_count = sum(word.lower() in text.lower().split() for word in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Prever a resposta
def predict_answer(conversation, question, choices):

    conversation_sentiment = analyze_sentiment(conversation)

    scores = [0] * len(choices)
    
    # Regra 1: Bigramas
    conversation_bigrams = set(extract_bigrams(conversation))
    question_bigrams = set(extract_bigrams(question))
    choices_bigrams = [set(extract_bigrams(choice)) for choice in choices]
    for i, choice_bigrams in enumerate(choices_bigrams):
        scores[i] += len(conversation_bigrams.intersection(choice_bigrams))
        scores[i] += len(question_bigrams.intersection(choice_bigrams))
    
    # Regra 2: Entidades
    conversation_entities = set(extract_entities(conversation))
    question_entities = set(extract_entities(question))
    choices_entities = [set(extract_entities(choice)) for choice in choices]
    for i, choice_entities in enumerate(choices_entities):
        scores[i] += len(conversation_entities.intersection(choice_entities))
        scores[i] += len(question_entities.intersection(choice_entities))

    # Regra 3: Sentimento
    for i, choice in enumerate(choices):
        choice_sentiment = analyze_sentiment(choice)
        # Adicionar score se o sentimento da opção de resposta corresponder ao sentimento
        scores[i] += (conversation_sentiment == choice_sentiment)
        # Adicionar score se o sentimento da opção de resposta corresponder ao sentimento da pergunta
        question_sentiment = analyze_sentiment(question)
        scores[i] += (question_sentiment == choice_sentiment)
 
    # Regra 4: Funções Gramaticais e Relações
    # TODO: adicionar as funções gramaticais e relações

    # Prever a opção de resposta com a pontuação mais alta
    predicted_idx = scores.index(max(scores))
    return choices[predicted_idx]


# Funções necessárias para o SAS
# TODO: verificar se isto tá certo
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

    # Exact Match
    exact_match = 0
    for (true, predicted) in zip(true_answer, predicted_answer):
        if true == predicted:
            exact_match += 1
    print("EXACT MATCH:", exact_match)

    # F1-score
    f1 = f1_score(true_answer, predicted_answer, average='weighted')
    print("F1-SCORE:", f1)

    # Similaridade - uso do coseno  ##### POR CONFIRMAR
    tf_vect = TfidfVectorizer(ngram_range=(1, 3), strip_accents='unicode', max_features=500, min_df=5, max_df=0.75)

    true = tf_vect.fit_transform(true_answer)
    predicted = tf_vect.transform(predicted_answer)

    cosine_matrix = cosine_similarity(true, predicted)
    similarity = sum(cosine_matrix[i][i] for i in range(len(cosine_matrix)))
    print("SIMILARIDADE:", similarity)

    # Chamar a função do SAS
    sas_score = calculate_sas(true_answer, predicted_answer)
    print("SAS:", sas_score)