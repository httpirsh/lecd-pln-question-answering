import json
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nlp = spacy.load("en_core_web_sm")

def topic_modelling(conv, options):
    from sklearn.decomposition import LatentDirichletAllocation

    tf_vect = TfidfVectorizer(strip_accents='unicode')
    conv_vect = tf_vect.fit_transform([conv])

    lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.5, topic_word_prior=0.5)
    conv_topics = lda.fit_transform(conv_vect)

    opt_topics = [lda.transform(tf_vect.transform([option])) for option in options]

    similarity = [np.linalg.norm(conv_topics - opt_topics[i]) for i in range(len(options))]

    correct_option_index = max(range(len(options)), key=lambda i: similarity[i])
    correct_option = choices[correct_option_index]
    return correct_option


with open('train.json', 'r') as file:
    data = json.load(file)

true_answers = []
predicted_answers = []

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

        true_answers.append(answer)
        pred_answer = topic_modelling(conversation_text, choices)
        predicted_answers.append(pred_answer)
