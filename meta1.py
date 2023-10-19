import json
import spacy
import re

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
    choices_bigrams = [set(extract_bigrams(choice)) for choice in choices]
    for i, choice_bigrams in enumerate(choices_bigrams):
        scores[i] += len(conversation_bigrams.intersection(choice_bigrams))
    
    # Regra 2: Entidades
    conversation_entities = set(extract_entities(conversation))
    choices_entities = [set(extract_entities(choice)) for choice in choices]
    for i, choice_entities in enumerate(choices_entities):
        scores[i] += len(conversation_entities.intersection(choice_entities))
    
    # Regra 3: Sentimento
    for i, choice in enumerate(choices):
        choice_sentiment = analyze_sentiment(choice)
        # Adicionar pontuação se o sentimento da opção de resposta corresponder ao sentimento 
        scores[i] += (conversation_sentiment == choice_sentiment)
 
    # Regra 4: Funções Gramaticais e Relações
    # ......

    # Prever a opção de resposta com a pontuação mais alta
    predicted_idx = scores.index(max(scores))
    return choices[predicted_idx]

# Testar com uma amostra
sample_conversation_text = " ".join(data[0][0])
sample_qa_pair = data[0][1][0]
sample_question = sample_qa_pair['question']
sample_choices = sample_qa_pair['choice']
sample_answer = sample_qa_pair['answer']

predicted_answer = predict_answer(sample_conversation_text, sample_question, sample_choices)
print(f"Predicted: {predicted_answer}, Actual: {sample_answer}")

# Testar tudo e fazer previsões
correct_predictions = 0
total_questions = 0

for conversation_data in data:
    conversation_text = " ".join(conversation_data[0])
    
    for qa_pair in conversation_data[1]:
        question = qa_pair['question']
        choices = qa_pair['choice']
        answer = qa_pair['answer']
        
        predicted_answer = predict_answer(conversation_text, question, choices)
        if predicted_answer == answer:
            correct_predictions += 1
        
        total_questions += 1

accuracy = correct_predictions / total_questions if total_questions > 0 else 0
(correct_predictions, total_questions, accuracy)

