from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
from nltk.corpus import wordnet as wn
from transformers import pipeline
from statistics import mean

import pdb


model_name = "deepset/roberta-base-squad2"
model_names = ["deep-learning-analytics/triviaqa-t5-base",
               "google/t5-11b-ssm-tqa",
               "google/t5-3b-ssm",
               "google/t5-small-ssm-nq",
               ]
precomputed_embs = dict()


def context_qa(question, context):
    """

    :param question:
    :param context:
    :return:
    """
    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res


def open_qa(question):
    """

    :param question:
    :return:
    """
    #     nlp = pipeline('question-answering', model=model_names[0], tokenizer=model_name)
    #     QA_input = {
    #         'question': question,
    #     }
    #     res = nlp(QA_input)

    # TODO look for other text-generation pipelines
    gen = pipeline('text-generation', model='EleutherAI/gpt-neo-125M', tokenizer='EleutherAI/gpt-neo-125M')
    res = gen(question, max_length=32)

    return res


def load_roberta_tokenizer():
    """

    :return:
    """
    return RobertaTokenizer.from_pretrained('roberta-base')


def load_roberta_model():
    """

    :return:
    """
    return RobertaModel.from_pretrained('roberta-base')


def embed_text(text, roberta_tokenizer=None, roberta_model=None):
    """

    :param text:
    :param roberta_tokenizer:
    :param roberta_model:
    :return:
    """
    if roberta_tokenizer is None:
        roberta_tokenizer = load_roberta_tokenizer()
    if roberta_model is None:
        roberta_model = load_roberta_model()
    encoded_input = roberta_tokenizer(text, return_tensors='pt')
    output = roberta_model(**encoded_input)
    return output.pooler_output


def text_similarity(text_1, text_2):
    """Returns a similarity degree between two texts"""
    if text_1 in precomputed_embs.keys():
        enc_1 = precomputed_embs[text_1]
    else:
        enc_1 = embed_text(text_1)
        precomputed_embs[text_1] = enc_1

    if text_2 in precomputed_embs.keys():
        enc_2 = precomputed_embs[text_2]
    else:
        enc_2 = embed_text(text_2)
        precomputed_embs[text_2] = enc_2

    return cosine_similarity(enc_1, enc_2)


def get_definitions(word: str,
                    num_definitions: int = None):
    """

    :param word:
    :param num_definitions:
    :return:
    """
    definitions = [synset.definition() for synset in wn.synsets(word)]
    if num_definitions:
        definitions = definitions[:num_definitions]
    return ".".join(definitions)


def main():
    labels = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
    labels_contexts = dict()
    for label in labels:
        labels_contexts = get_definitions(label)

    pos_label_texts = ["yes", "definitely", "of course", "undoubtedly",
                       "do not agree", "perhaps", "maybe", "certainly"]
    neg_label_texts = ["no", "not", "do not agree", "impossible",
                       "can't", "couldn't", "nope", ]

    sentences = ["women are stupid", "men and women are equal",
                 "men are superior", "tomorrow will rain"]

    ground_truth = [1, 0, 1, 0]

    print(ground_truth)

    for label in labels:
        for sentence in sentences:
            question = f"Is '{sentence}' {label}?"
            current_context = labels_contexts

            context_answer = context_qa(question=question, context=current_context)
            open_answer = open_qa(question=question)

            context_pos_sims = [text_similarity(context_answer, text) for text in pos_label_texts]
            context_neg_sims = [text_similarity(context_answer, text) for text in neg_label_texts]
            context_label = int(mean(context_neg_sims) < mean(context_pos_sims))

            open_pos_sims = [text_similarity(open_answer, text) for text in pos_label_texts]
            open_neg_sims = [text_similarity(open_answer, text) for text in neg_label_texts]
            open_label = int(mean(open_neg_sims) < mean(open_pos_sims))

            print("*" * 20)
            print(question)
            print("Context QA answer:", context_answer)
            print("Open QA answer:", open_answer)
            print("Context QA label:", context_label)
            print("Open QA label:", open_label)
            print("*" * 20)


if __name__ == "__main__":
    main()
