import csv
import linalg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
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

from src.keras_model import keras_model

from src.keras_model import *
from src.text_preprocess import *
from src.xgb_model import *
from utils import *

from src.embeddings_train.train_word2vec import document_preprocess

stop_words = set(stopwords.words('english'))



def mask_expression(text, start_offset, end_offset):
    return text[:start_offset] + "<mask>" + text[end_offset:]


def predict_masked_tokens(text):
    unmasker = pipeline('fill-mask', model='roberta-base')
    return unmasker(text)[0]["token_str"]

def main():
    model_option = ["KERAS_MODEL", "XGBREGRESSOR"][-1]

    all_embedding_features = ["phrase", "target_word", "phrase_with_no_target_word", "predicted_target_word", "predicted_phrase", "phrase_special_tokenized"][0]

    embedding_feature = "target_word"
    embedding_model = "paper_features" #"tfidfvectorizer_char"

    embedding_models = ["tfidfvectorizer_char", "tfidfvectorizer_char_wb"]
    embedding_features = ["target_word", "target_word"]

    use_multiple_models = False
    new_embedding = True

    if new_embedding:
        use_loaded_data = False
    else:
        use_loaded_data = True

    print("Embedding model:", embedding_model)
    if use_multiple_models == False:
        if use_loaded_data:
            X_train, y_train, X_test = load_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
            if len(X_train.shape) == 1:
                X_train = np.reshape(X_train, newshape=(X_train.shape[0], 1))
                X_test = np.reshape(X_test, newshape=(X_test.shape[0], 1))
        else:
            X_train, y_train, X_test = embed_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
    else:
        if use_loaded_data:
            strategy = "stacking"
            embedding_model = "_".join(embedding_models) + "_" + strategy
            X_train, y_train, X_test = load_multiple_models(embedding_models=embedding_models, embedding_features=embedding_features, strategy=strategy)
        else:
            X_train, y_train, X_test = embed_multiple_models(embedding_models=embedding_models, embedding_features=embedding_features, strategy=strategy)

    if model_option == "KERAS_MODEL":
        if embedding_model == "word2vec_trained" or embedding_model == "word2vec_trained_special":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[-1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[-1]))
        labels = keras_model(X_train, y_train, X_test)
    elif model_option == "XGBREGRESSOR":
        if embedding_model == "word2vec_trained" or embedding_model == "word2vec_trained_special":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[-1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[-1]))
        labels = train_basic_model(X_train, y_train, X_test, embedding_feature=embedding_feature, embedding_model=embedding_model)
    else:
        raise Exception("Unknown model option")

    ids = list(range(14003, 15767))
    create_submission_file(ids, labels)


if __name__ == '__main__':
    main()
