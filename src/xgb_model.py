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
import pdb
"""
0.05356
0.05679756082963418 10 1
BEST MAX DEPTH: ******** 10
BEST MIN CHILD WEIGHT: ******** 1


0.0577148782946343 0.5
BEST ALPHA: ******** 0.3
0.05770197753634105 1 1
BEST SUBSAMPLE: ******** 1
BEST COSAMPLE BYTREE: ******** 1

target:

num_characters, num_vowels, num_consonants
%_characters, %_vowels DA DA
num_double_consonants as % of total num of letters DA
n_grams of 1,2,3,4 characters DA DA DA DA

part of speech

number of senses in wordnet (summed if multiple words) 

context:
min, max and mean for the cosine similarity of the target and each other word from the sentence for word2vec embeddings
same from 14 for sense embeddings

Embedding feature: target_word
Embedding model: paper_features
********************
TEST RESULT:  0.05910654819095028 with all 5 hyper param optimizations
********************


"""
from finetune_xgb import finetune_xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor

def train_basic_model(X_train, y_train, X_test, embedding_feature: str = "target_word", embedding_model: str = "roberta"):
    data_scaler = StandardScaler()
    label_scaler = StandardScaler()
    regressor = XGBRegressor(eval_metric=mean_absolute_error,  max_depth=10, min_child_weight=1) #alpha=0.3, subsample=1, cosample_bytree=1,)
    regressor = SVR() # 0.07623
    regressor = DecisionTreeRegressor() # 0.06727
    regressor = AdaBoostRegressor() # 0.10
    regressor = GradientBoostingRegressor() # 0.065
    regressor = HuberRegressor() # 0.086
    regressor = SGDRegressor() # 0.10
    regressor = LinearRegression() # 0.10
    regressor = MLPRegressor() #
    regressor = BayesianRidge()
    regressor = RandomForestRegressor(random_state=100)  # 0.0543

    from sklearn.model_selection import KFold

    def cross_val_func(regressor, X_train, y_train):
        kfolder5 = KFold(n_splits=5, shuffle=False)
        print(X_train.shape)
        scores = cross_val_score(regressor, X_train, y_train, scoring='neg_mean_absolute_error', cv=kfolder5, n_jobs=-1)
        return [score * (-1) for score in scores]

    regressors_list = [RandomForestRegressor(random_state=100), LinearRegression(), SVR()]
    regressor_names = ["random forest", "linear regressor", "svr"]
    for (regressor, name) in list(zip(regressors_list, regressor_names)):
        print(name,":",cross_val_func(regressor, X_train, y_train))
    """
    Results of cross validation:
    random forest : [0.062240533679206164, 0.05939914829907854, 0.061634080984518023, 0.05774259217534336, 0.06274576765079065]
    linear regressor : [0.1117310780087325, 0.10811122489629153, 0.10634572244232848, 0.11075799032015514, 0.11112644831981833]
    svr : [0.12918072212066153, 0.12330485614172973, 0.12103617748228726, 0.12373396212906405, 0.1251740540981331]
    """
    # 0.05169
    print(X_train.shape, X_test.shape, X_train.dtype, X_test.dtype)

    X_train = data_scaler.fit_transform(X_train)
    X_test = data_scaler.transform(X_test)

    y_train = label_scaler.fit_transform(np.reshape(y_train, (y_train.shape[0], 1)))

    regressor.fit(X_train, y_train)
    print("Embedding feature:", embedding_feature)
    print("Embedding model:", embedding_model)

    # finetune_xgb(X_train, y_train, X_test, label_scaler)
    # y_pred = np.clip(y_pred, a_min=0.0, a_max=1.0)
    # print("MAE:",  mean_absolute_error(y_validation, y_pred))
    return label_scaler.inverse_transform(regressor.predict(X_test))


"""
SVR: TEST RESULT:  0.07623491324164598
XGBR: TEST RESULT:  0.05770197753634105

"""