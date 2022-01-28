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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from typing import List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def keras_model(X_train, y_train, X_test):
    n_hiddens = 512
    input_dimension = X_train[0].shape[0]
    num_epochs = 2
    batch_size = 32
    n_outputs = 1

    # defining the regressor
    regressor = Sequential()
    regressor.add(Dense(n_hiddens, input_dim=input_dimension, activation='relu'))
    regressor.add(Dense(n_hiddens, activation='relu'))
    regressor.add(Dense(n_hiddens, activation='relu'))
    regressor.add(Dense(n_hiddens, activation='relu'))
    regressor.add(Dense(n_outputs))
    
    # compiling the regressor
    regressor.compile(loss='mean_absolute_error', optimizer='adam')

    # fitting the regressor
    regressor.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)


    labels = regressor.predict(X_test)

    # print(labels.shape)
    # print(labels[0])
    # print(labels[-1])

    # exit(0)
    return labels


"""
ST RESULT:  0.08734544425405025 -> keras

"""