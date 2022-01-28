import csv
import linalg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pdb
import torch
import pdb
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


numpy_arrays_path = "data/numpy_data"

def create_submission_file(ids, labels, imputer_strategy: str = "max"):
    submission_file_path = "data/submission.txt"

    with open(submission_file_path, 'r+') as f:
        f.truncate(0)

    with open(submission_file_path, "a") as f:
        f.write("id,label\n")
        for id, label in zip(ids, labels):
            if label < 0.0:
                if imputer_strategy == "max":
                    label = max(0.0, label)
                elif imputer_strategy == "mean":
                    label = np.mean(labels)
                elif imputer_strategy == "median":
                    label = np.median(labels)
                else:
                    raise Exception("Unknown imputer strategy")
            f.write(str(id) + "," + str(label) + "\n")

def load_data(embedding_feature: str = "target_word", embedding_model: str = "roberta"):
    X_train = np.load(file=os.path.join(numpy_arrays_path, "X_train_" + embedding_feature + "_" + embedding_model + ".npy"), allow_pickle=True)
    y_train = np.load(file=os.path.join(numpy_arrays_path, "y_train_" + embedding_feature + "_" + embedding_model + ".npy"), allow_pickle=True)
    X_test = np.load(file=os.path.join(numpy_arrays_path, "X_test_" + embedding_feature + "_" + embedding_model + ".npy"), allow_pickle=True)

    y_train = y_train.astype("float32")

    # import pdb
    # pdb.set_trace()
    return X_train, y_train, X_test


def load_multiple_models(embedding_models: List[str], embedding_features: List[str], strategy: str = "averaging"):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    for (embedding_model, embedding_feature) in zip(embedding_models, embedding_features):
        X_train, y_train, X_test = load_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)

    # print([x.shape for x in X_train_list])
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

