import json
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")


with open('train.json', 'r') as file:
    data = json.load(file)

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
                                 if conversation_text[i].startswith("W:") or conversation_text[i].startswith("F:")]

        elif (len(man_in_question) > 0 and len(woman_in_question) == 0
              and dep_question["man"] == "nsubj"):
            conversation_text = [conversation_text[i][conversation_text[i].index(":")+2:] for i in
                                 range(len(conversation_text))
                                 if conversation_text[i].startswith("M:")]

        else:
            conversation_text = [conversation_text[i][conversation_text[i].index(":")+2:] for i in
                                 range(len(conversation_text))]

        conversation_text = " ".join(conversation_text)
        nlp_conv = nlp(conversation_text)

        # USAR TF-IDF PARA REPRESENTAR OS DADOS E FAZER MODELOS (ou regras, por confirmar)
        # REPETIR PARA OUTRO MÉTODO DE REPRESENTAÇÃO E VER QUAL É O MELHOR