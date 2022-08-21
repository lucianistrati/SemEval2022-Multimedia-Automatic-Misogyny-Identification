from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.tokenize import TreebankWordTokenizer as twt
from wordsegment import load as load_wordsegment
from sklearn.metrics import mean_absolute_error
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from statistics import mean, median
from wordfreq import word_frequency
from collections import defaultdict
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from wordsegment import segment
from textblob import TextBlob
from sklearn.svm import SVR
from pandas import read_csv
from gensim import corpora
from copy import deepcopy
from tqdm import tqdm

import numpy as np

import pronouncing
import textstat
import inflect
import linalg
import string
import spacy
import copy
import nltk
import csv
import pdb
import os

textstat.set_lang("en")
load_wordsegment()
PAD_TOKEN = "__PAD__"
money_symbols = ["$", "£", "€", "lei", "RON", "USD", "EURO", "dolari", "lire", "yeni"]
roman_numerals = "XLVDCMI"
GOOD = 0
ERRORS = 0
LEFT_LEFT_TOKEN = -4
LEFT_TOKEN = -3
RIGHT_TOKEN = -1
RIGHT_RIGHT_TOKEN = -2
numpy_arrays_path = "data/numpy_data"
# word2vec_model = Word2Vec.load("src/embeddings_train/fasttext.model")


def load_inflection_engine():
    """

    :return:
    """
    return inflect.engine()


def load_word2vec_model():
    """

    :return:
    """
    return Word2Vec.load("checkpoints/word2vec.model")


def load_stopwords():
    """"""
    return set(stopwords.words("english"))


def load_lemmatizer():
    """

    :return:
    """
    return WordNetLemmatizer()


def load_stemmer():
    """

    :return:
    """
    return PorterStemmer()


def load_all_languages():
    """

    :return:
    """
    all_languages = set(list(np.load(file="data/massive_all_4749_languages.npy", allow_pickle=True)))
    return all_languages


def load_nlp():
    """

    :return:
    """
    nlp = spacy.load("ro_core_news_sm")
    return nlp


def load_all_stopwords(nlp):
    """

    :param nlp:
    :return:
    """
    all_stopwords = set(list(nlp.Defaults.stop_words))
    return all_stopwords


def is_there_a_language(text, all_languages: List[str]):
    """

    :param text:
    :param all_languages:
    :return:
    """
    for lang in all_languages:
        if lang in text:
            return True
    return False


def might_be_feminine_surname(text):
    """

    :param text:
    :return:
    """
    text = text.lower()
    return text.endswith("ei") or text.endswith("a")


def get_stopwords_pct(text, all_stopwords: List[str]):
    """

    :param text:
    :param all_stopwords:
    :return:
    """
    tokens = set(word_tokenize(text))
    return len(tokens.intersection(all_stopwords)) / len(tokens)


def count_letters(target):
    """

    :param target:
    :return:
    """
    return len(target)


def get_phrase_len(phrase):
    """

    :param phrase:
    :return:
    """
    return len(phrase)


def get_num_pos_tags(sentence, tokens=None):
    """

    :param sentence:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(sentence) if tokens is None else tokens
    pos_tags = nltk.pos_tag(tokens)
    pos_tags = [pos_tag[1] for pos_tag in pos_tags]
    return len(set(pos_tags)) / len(tokens)


def get_word_position_in_phrase(phrase, start_offset):
    """

    :param phrase:
    :param start_offset:
    :return:
    """
    return start_offset / len(phrase)


def get_phrase_num_tokens(phrase):
    """

    :param phrase:
    :return:
    """
    return len(word_tokenize(phrase))


def has_money_tag(text):
    """

    :param text:
    :return:
    """
    global money_symbols
    for sym in money_symbols:
        if sym.lower() in text.lower():
            return True
    return False


def starts_with_capital_letter(word):
    """

    :param word:
    :return:
    """
    if word[0] in string.ascii_uppercase:
        return True
    return False


def get_len(text):
    """

    :param text:
    :return:
    """
    return len(text)


def get_capital_letters_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c in string.ascii_uppercase]) / len(text)


def get_roman_numerals_pct(text):
    """

    :param text:
    :return:
    """
    global roman_numerals
    return len([c for c in text if c in roman_numerals]) / len(text)


def get_digits_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c in string.digits]) / len(text)


def get_punctuation_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c in string.punctuation]) / len(text)


def get_dash_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c == "-"]) / len(text)


def get_spaces_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c == " "]) / len(text)


def get_slashes_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c == "/" or c == "\\"]) / len(text)


def get_text_similarity(text_1, text_2):
    """

    :param text_1:
    :param text_2:
    :return:
    """
    pass


def get_dots_pct(text):
    """

    :param text:
    :return:
    """
    return len([c for c in text if c == "."]) / len(text)


def count_capital_words(text, tokens=None):
    """

    :param text:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(text) if tokens is None else tokens
    return sum(map(str.isupper, tokens))


def count_punctuations(text):
    """

    :param text:
    :return:
    """
    punctuations = """}!'#/$%"(*]+,->.:);=?&@\^_`{<|~["""
    res = []
    for i in punctuations:
        res.append(text.count(i))
    if len(res):
        return mean(res)
    return 0.0


def get_word_frequency(target, tokens=None):
    """

    :param target:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(target) if tokens is None else tokens
    return mean([word_frequency(token, "ro") for token in tokens])


def count_sws(text, tokens=None, stop_words=List[str]):
    """

    :param text:
    :param tokens:
    :param stop_words:
    :return:
    """
    if tokens is None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    """

    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


encoder_dict = dict()
encoder_cnt = 0


def get_pos_tags(token, doc, nlp_doc, index):
    """

    :param token:
    :param doc:
    :param nlp_doc:
    :param index:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(nlp_doc))))
    context_tokens = [tok for i, tok in enumerate(word_tokenize(doc)) if i in context_indexes]
    print(context_tokens)
    feats = []
    for idx, nlp_token in enumerate(nlp_doc):
        if idx in context_indexes:
            feats.append(nlp_token.pos_)
    if index == 1 or index == len(nlp_doc) - 2:
        feats.insert(0, "-2_pos")
        feats.append("-3_pos")
    if index == 0 or index == len(nlp_doc) - 1:
        feats.insert(0, "-2_pos")
        feats.insert(0, "-1_pos")
        feats.append("-3_pos")
        feats.append("-4_pos")
    return feats


def get_dep_tags(token, doc, nlp_doc, index):
    """

    :param token:
    :param doc:
    :param nlp_doc:
    :param index:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(nlp_doc))))
    context_tokens = [tok for i, tok in enumerate(word_tokenize(doc)) if i in context_indexes]
    print(context_tokens)
    feats = []
    for idx, nlp_token in enumerate(nlp_doc):
        if idx in context_indexes:
            feats.append(nlp_token.dep_)
    if index == 1 or index == len(nlp_doc) - 2:
        feats.insert(0, "-2_dep")
        feats.append("-3_dep")
    if index == 0 or index == len(nlp_doc) - 1:
        feats.insert(0, "-2_dep")
        feats.insert(0, "-1_dep")
        feats.append("-3_dep")
        feats.append("-4_dep")
    return feats


def get_ner_tags(token, doc, nlp_doc, index):
    """

    :param token:
    :param doc:
    :param nlp_doc:
    :param index:
    :return:
    """
    global encoder_dict, encoder_cnt
    context_indexes = list(range(max(index - 2, 0), min(index + 2, len(nlp_doc))))
    context_tokens = [tok for i, tok in enumerate(word_tokenize(doc)) if i in context_indexes]

    feats = []
    for nlp_token in nlp_doc.ents:
        if nlp_token.text in context_tokens:
            feats.append(nlp_token.label_)
    if index == 1 or index == len(nlp_doc) - 2:
        feats.insert(0, "-2_ner")
        feats.append("-3_ner")
    if index == 0 or index == len(nlp_doc) - 1:
        feats.insert(0, "-2_ner")
        feats.insert(0, "-1_ner")
        feats.append("-3_ner")
        feats.append("-4_ner")

    return feats


def get_paper_features(token, document, index, nlp):
    """

    :param token:
    :param document:
    :param index:
    :param nlp:
    :return:
    """
    nlp_doc = nlp(document)
    doc = document
    # import pdb
    # pdb.set_trace()
    linguistical_features = [get_sws_pct(token), count_sws(token), get_dots_pct(token), get_dash_pct(token),
                             get_len(token), get_digits_pct(token), get_punctuation_pct(token), get_phrase_len(document),
                             get_spaces_pct(token), get_capital_letters_pct(token), get_slashes_pct(token), index,
                             get_roman_numerals_pct(token), get_stopwords_pct(token)]

    pos_tags = get_pos_tags(token, doc, nlp_doc, index)
    dep_tags = get_dep_tags(token, doc, nlp_doc, index)
    ner_tags = get_ner_tags(token, doc, nlp_doc, index)
    string_feats = pos_tags + dep_tags + ner_tags

    return np.array(linguistical_features), " ".join(string_feats)


def main():
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets("flexed"):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())


if __name__ == "__main__":
    main()
