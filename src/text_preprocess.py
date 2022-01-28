import csv
import linalg
import matplotlib.pyplot as plt
import nltk

import textstat

textstat.set_lang("en")


import numpy as np
import os
import pdb
import pdb
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from gensim import corpora
from gensim.models import Word2Vec
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from transformers import pipeline
from typing import List
from xgboost import XGBRegressor

from src.feature_extractor import *

PAD_TOKEN = "__PAD__"
word2vec_model = Word2Vec.load("src/embeddings_train/word2vec.model")

numpy_arrays_path = "data/numpy_data"
# word2vec_model = Word2Vec.load("src/embeddings_train/fasttext.model")

import copy
from src.embeddings_train.train_word2vec import document_preprocess


# def document_preprocess(document):
#     return document.lower().split()

# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.syn1neg.npy")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.wv.vectors.npy")


def embed_text(text, embedding_model: str = "word2vec_trained", phrase="", start_offset="", end_offset=""):
    """This function embedds the text given using an embedding model, it might also use the phrase, start_offset or the end_offset for certain embeddings"""
    # print(embedding_model)
    if embedding_model == "roberta":
        encoded_input = roberta_tokenizer(text, return_tensors='pt')
        output = roberta_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "sentence_transformer":
        embedding = sent_transf_model.encode(text)
        return embedding
    elif embedding_model == "sentence_transformer_multi_qa":
        embedding = sent_transf_model_multi_qa.encode(text)
        return embedding
    elif embedding_model == "roberta_large":
        encoded_input = roberta_large_tokenizer(text, return_tensors='pt')
        output = roberta_large_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "bert_large":
        encoded_input = bert_large_tokenizer(text, return_tensors='tf')
        output = bert_large_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "albert-base-v2":
        encoded_input = albert_tokenizer(text, return_tensors='tf')
        output = albert_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "roberta_distil":
        embedding = sent_transf_model_distil_roberta.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l12":
        embedding = all_minimlm_l12_model.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l6":
        embedding = all_minimlm_l6_model.encode(text)
        return embedding
    elif embedding_model.startswith("tfidfvectorizer"):
        return text
    elif embedding_model.startswith("gensim_doc2bow"):
        return text.split()
    elif embedding_model == "word2vec_trained":
        try:
            vector = word2vec_model.wv[document_preprocess(text)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector
    elif embedding_model == "paper_features":
        return get_paper_features(phrase, text, start_offset, end_offset), None
    elif embedding_model == "word2vec_trained_special":
        try:
            vector = word2vec_model.wv[text]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector


def embed_data(embedding_feature: str, embedding_model: str):
    """This function embedds the data based on a chosen feature to be emebedded/chosen set of rules to be applied over the feature with a certain alias
    and then an embedding model might be used to pursue the feature extraction"""
    train_path = "data/train_full.txt"
    test_path = "data/test.txt"
    if embedding_feature == "phrase":
        column_idx = 1
    elif embedding_feature == "target_word":
        column_idx = 4
    # else:
    #     raise Exception("Wrong embedding feature!")

    train_columns = ["id", "phrase", "start_offset", "end_offset", "target_word",
                     "native_annotators", "non_native_annotators", "difficult_native_annotators",
                     "difficult_non_native_annotators", "label"]
    print("Embedding feature:", embedding_feature)
    test_columns = ["id", "phrase", "start_offset", "end_offset", "target_word",
                    "native_annotators", "non_native_annotators"]

    X_train, y_train, X_test = [], [], []

    datapoints_limit = -1

    X_train_str = []
    X_test_str = []
    with open(train_path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        for i, row in tqdm(enumerate(data)):
            if datapoints_limit != -1:
                if i > datapoints_limit:
                    break
            if embedding_feature == "phrase_with_no_target_word":
                start_offset = int(row[2])
                end_offset = int(row[3])
                # print(row[1])
                # import pdb
                # pdb.set_trace()
                X_train.append(embed_text(row[1][:start_offset] + row[1][end_offset:], embedding_model=embedding_model))
            elif embedding_feature in ["phrase", "target_word"]:
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                outputs = embed_text(row[column_idx], embedding_model=embedding_model, phrase=phrase, start_offset=start_offset, end_offset=end_offset)
                # vecs = []
                # res_types = []
                # final_vecs = []
                # for word in word_tokenize(row[column_idx]):
                #     res_type, vec = get_embedding_word(word)
                #     res_types.append(res_type)
                #     vecs.append(vec)
                # if "good" in res_types:
                #     for res_type in res_types:
                #         if res_type == "good":
                #             final_vecs.append(vec)
                # if len(final_vecs):
                #     vecs = final_vecs
                # vecs = np.mean(vecs, axis=0)
                # assert vecs.shape == (1,10)
                if embedding_model == "paper_features":
                    # for elem in vecs[0]:
                    #     outputs[0].append(elem)
                    # print(np.array(outputs[0]).shape)
                    # print(np.array(outputs[0]).astype(float))
                    X_train.append(np.array(outputs[0]))
                    # print(np.array(X_train).shape)
                    if outputs[1] is not None:
                        # print(outputs[1], type(outputs[1]))
                        X_train_str.append(outputs[1])
                else:
                    X_train.append(outputs)
            elif embedding_feature == "predicted_target_word":
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                masked_phrase = mask_expression(phrase, start_offset, end_offset)
                masked_prediction = predict_masked_tokens(masked_phrase)
                X_train.append(embed_text(masked_prediction, embedding_model=embedding_model))
            elif embedding_feature == "phrase_special_tokenized":
                start_offset = int(row[2])
                end_offset = int(row[3])
                X_train.append(embed_text(row[1][:start_offset].lower().split() + [row[4].lower()] + row[1][end_offset:].lower().split(), embedding_model=embedding_model))
            y_train.append(float(row[8]) / float(row[6]))

    X_train, y_train = np.array(X_train), np.array(y_train)

    np.save(file=os.path.join(numpy_arrays_path, "y_train_" + embedding_feature + "_" + embedding_model + "_non_native.npy"), arr=y_train)

    with open(test_path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        for i, row in tqdm(enumerate(data)):
            if datapoints_limit != -1:
                if i > datapoints_limit:
                    break
            if embedding_feature == "phrase_with_no_target_word":
                start_offset = int(row[2])
                end_offset = int(row[3])
                X_test.append(embed_text(row[1][:start_offset] + row[1][end_offset:], embedding_model=embedding_model))
            elif embedding_feature in ["phrase", "target_word"]:
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                outputs = embed_text(row[column_idx], embedding_model=embedding_model, phrase=phrase, start_offset=start_offset, end_offset=end_offset)
                # vecs = []
                # res_types = []
                # final_vecs = []
                # for word in word_tokenize(row[column_idx]):
                #     res_type, vec = get_embedding_word(word)
                #     res_types.append(res_type)
                #     vecs.append(vec)
                # if "good" in res_types:
                #     for res_type in res_types:
                #         if res_type == "good":
                #             final_vecs.append(vec)
                # if len(final_vecs):
                #     vecs = final_vecs
                # vecs = np.mean(vecs, axis=0)
                # assert vecs.shape == (1,10)
                if embedding_model == "paper_features":
                    # for elem in vecs[0]:
                    #     outputs[0].append(elem)
                    X_test.append(np.array(outputs[0], dtype=np.float))
                    if outputs[1] is not None:
                        X_test_str.append(outputs[1])
                else:
                    X_test.append(outputs)
            elif embedding_feature == "predicted_target_word":
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                masked_phrase = mask_expression(phrase, start_offset, end_offset)
                masked_prediction = predict_masked_tokens(masked_phrase)
                # print(masked_prediction)
                X_test.append(embed_text(masked_prediction, embedding_model=embedding_model))
            elif embedding_feature == "phrase_special_tokenized":
                # print(row[4].lower())
                X_test.append(embed_text(row[1][:start_offset].lower().split() + [row[4].lower()] + row[1][end_offset:].lower().split(), embedding_model=embedding_model))

    X_test = np.array(X_test)

    # corpus = [x for x in X_train] + [x for x in X_test]
    # corpus = list(set(corpus))
    # print(len(corpus))
    # print(corpus[0])
    # print(corpus[-1])
    # np.save(file="data/wce_dataset.npy", arr=np.array(corpus), allow_pickle=True)
    # exit(0)

    print(X_train.shape, X_test.shape, "PRE CONCAT WITH STR FEATURES")
    if len(X_test_str) and len(X_train_str):
        cv = TfidfVectorizer(analyzer='char')
        X_train_str = cv.fit_transform(X_train_str).toarray()
        X_test_str = cv.transform(X_test_str).toarray()

        print(X_train_str.shape, X_test_str.shape)

        X_train_str = X_train_str.astype(float)
        X_test_str = X_test_str.astype(float)

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        print(X_train_str.dtype)
        print(X_test_str.dtype)

        print(X_train.dtype)
        print(X_test.dtype)

        X_train = np.hstack((X_train, X_train_str))
        X_test = np.hstack((X_test, X_test_str))

    print(X_train.shape, X_test.shape, "POST CONCAT WITH STR FEATURES")

    tfidfvectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4))

    if embedding_model.startswith("tfidfvectorizer"):
        X_train = tfidfvectorizer.fit_transform(X_train).toarray()
        X_test = tfidfvectorizer.transform(X_test).toarray()
    elif embedding_model == "gensim_doc2bow":
        corpus = [x for x in X_train] + [x for x in X_test]
        import pdb
        dict_d = corpora.Dictionary(corpus)

        X_train = [dict_d.doc2bow(x) for x in X_train]  # one list of tuples of two integers
        X_train = np.array([np.array(x) for x in X_train])
        X_train = np.array([np.linalg.norm(x) for x in X_train])  # one float per sentence/word

        X_test = [np.array(dict_d.doc2bow(x)) for x in X_test]  # one list of tuples of two integers
        X_test = np.array([np.array(x) for x in X_test])
        X_test = np.array([np.linalg.norm(x) for x in X_test])  # one float per sentence/word

    np.save(file=os.path.join(numpy_arrays_path, "X_train_" + embedding_feature + "_" + embedding_model + ".npy"), arr=X_train)
    np.save(file=os.path.join(numpy_arrays_path, "y_train_" + embedding_feature + "_" + embedding_model + ".npy"), arr=y_train)
    np.save(file=os.path.join(numpy_arrays_path, "X_test_" + embedding_feature + "_" + embedding_model + ".npy"), arr=X_test)

    return X_train, y_train, X_test


from nltk.corpus import wordnet


def count_word_senses(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(len(wordnet.synsets(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_vowels(word):
    return len([c for c in word if c in "aeiou"])


def count_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    return len([c for c in word if c in consonants])


def count_double_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    cnt = 0
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i] in consonants and word[i + 1] in consonants:
            cnt += 1
    return cnt


def get_double_consonants_pct(word):
    return count_double_consonants(word) / len(word)


def get_vowel_pct(word):
    return count_vowels(word) / len(word)


def get_consonants_pct(word):
    return count_consonants(word) / len(word)


import nltk


def get_part_of_speech(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens == None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


from nltk.tokenize import TreebankWordTokenizer as twt


def spans(phrase):
    return list(twt().span_tokenize(phrase))


stop_words = set(stopwords.words('english'))


def count_sws(text, tokens=None):
    if tokens == None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


def get_context_tokens(phrase, start_offset, end_offset, context_size=1):  # try 2
    tokens = [PAD_TOKEN for _ in range(context_size)] + nltk.word_tokenize(phrase) + [PAD_TOKEN for _ in range(context_size)]
    tokens_spans = [(0, 0) for _ in range(context_size)] + spans(phrase) + [(0, 0) for _ in range(context_size)]
    for i, (l, r) in enumerate(tokens_spans):
        if r >= start_offset:
            return tokens[i - context_size: i] + tokens[i + 1: i + context_size + 1]
    return None


def embed_multiple_models(embedding_models: List[str], embedding_features: List[str], strategy: str = "averaging"):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    for (embedding_model, embedding_feature) in zip(embedding_models, embedding_features):
        X_train, y_train, X_test = embed_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)

    print([x.shape for x in X_train_list])
    # pdb.set_trace()

    if strategy == "averaging":
        X_train_list = np.array(X_train_list)
        X_test_list = np.array(X_test_list)
        X_train = np.mean(X_train_list, axis=0)
        y_train = y_train_list[0]
        X_test = np.mean(X_test_list, axis=0)
    elif strategy == "stacking":
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    elif strategy == "ensemble":
        return X_train_list, y_train_list, X_test_list
    elif strategy == "dimensionality_reduction":
        reduction_method = "PCA"  # "LDA", "TSNE"
        if reduction_method == "PCA":
            dimensionality_reducer = PCA()
        elif reduction_method == "LDA":
            dimensionality_reducer = LDA()
        elif reduction_method == "TSNE":
            dimensionality_reducer = TSNE()
        for (X_train, X_test) in zip(X_train_list, X_test_list):
            X_train = dimensionality_reducer.fit_transform(X_train, n_components=min(X_train.shape))
            X_test = dimensionality_reducer.transform(X_test, n_components=min(X_test.shape))
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    return X_train, y_train, X_test


GOOD = 0
ERRORS = 0


def get_paper_features(phrase, target, start_offset, end_offset):
    context_tokens = get_context_tokens(phrase, start_offset, end_offset)
    if context_tokens == None:
        context_tokens = []
    word = target
    global ERRORS, GOOD
    target_ = target

    # so far 0.057701 just with target and the 24 features
    num_features = []

    for target in [target_]:  # + context_tokens:
        word = target
        num_features_ = [count_letters(target),
                         count_consonants(target),
                         count_vowels(target),
                         get_vowel_pct(target),
                         get_consonants_pct(target),
                         get_double_consonants_pct(target),
                         count_word_senses(target, tokens=word_tokenize(target)),
                         mean([count_word_senses(context_tok) for context_tok in context_tokens]),
                         get_base_word_pct(target, tokens=word_tokenize(word)),
                         has_suffix(target, tokens=word_tokenize(word)),
                         count_letters(target),
                         get_base_word_pct_stem(target, tokens=word_tokenize(word)),
                         has_both_affixes_stem(target, tokens=word_tokenize(word)),
                         count_hypernyms(target, tokens=word_tokenize(word)),
                         count_hyponyms(target, tokens=word_tokenize(word)),
                         count_antonyms(target, tokens=word_tokenize(word)),
                         count_definitions_average_tokens_length(target, tokens=word_tokenize(word)),
                         count_definitions_average_characters_length(target, tokens=word_tokenize(word)),
                         count_definitions_tokens_length(target, tokens=word_tokenize(word)),
                         count_total_phonemes_per_pronounciations(target, tokens=word_tokenize(word)),
                         get_word_frequency(target, tokens=word_tokenize(word)),
                         get_average_syllable_count(target),
                         check_word_compounding(target),
                         get_base_word_pct(target),
                         ]
        for feature in num_features_:
            num_features.append(feature)
        test_data = target

        # num_features.append(textstat.flesch_reading_ease(test_data))
        # num_features.append(textstat.flesch_kincaid_grade(test_data))
        # num_features.append(textstat.smog_index(test_data))
        # num_features.append(textstat.coleman_liau_index(test_data))
        # num_features.append(textstat.automated_readability_index(test_data))
        # num_features.append(textstat.dale_chall_readability_score(test_data))
        # num_features.append(textstat.difficult_words(test_data))
        # num_features.append(textstat.linsear_write_formula(test_data))
        # num_features.append(textstat.gunning_fog(test_data))

        pos_features = None  # get_part_of_speech(target)

    """
    vectors = []
    for context_tok in context_tokens:
        if context_tok == PAD_TOKEN:
            continue
        try:
            # print(document_preprocess(context_tok))
            vector = word2vec_model.wv[document_preprocess(context_tok)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            GOOD += 1
        except KeyError:
             # continue
            vector = np.random.rand(1, 300)
            ERRORS += 1
        vectors.append(vector)

    import scipy
    try:
        # print(document_preprocess(context_tok))
        vector = word2vec_model.wv[document_preprocess(target)]
        vector = np.mean(vector, axis=0)
        vector = np.reshape(vector, (1, vector.shape[0]))
        GOOD += 1
    except KeyError:
        vector = np.random.rand(1, 300)
        ERRORS += 1

    target_vector = vector

    max_cos_sim = -1e18
    min_cos_sim = 1e18
    mean_cos_sim = 0.0

    for vector in vectors:
        ans = scipy.spatial.distance.cosine(np.reshape(vector, (vector.shape[-1], 1)), np.reshape(target_vector, (target_vector.shape[-1], 1)))
        max_cos_sim = max(max_cos_sim, ans)
        min_cos_sim = min(min_cos_sim, ans)
        mean_cos_sim += ans

    sum_cos_sim = copy.deepcopy(mean_cos_sim)
    mean_cos_sim /= len(vectors)
    """

    # num_features += [min_cos_sim, max_cos_sim, mean_cos_sim, sum_cos_sim]
    
    return num_features


if __name__ == '__main__':
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets('flexed'):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())
    # print(get_context_tokens(phrase, start_offset, end_offset))


