# regressor models
# https://www.programcreek.com/python/example/99826/xgboost.XGBRegressor

# https://www.kaggle.com/jayatou/xgbregressor-with-gridsearchcv
# https://www.kaggle.com/jayatou/xgbregressor-with-gridsearchcv
# https://www.kaggle.com/jayatou/xgbregressor-with-gridsearchcv
# https://codingdict.com/sources/py/xgboost/12190.html
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
from xgboost import XGBRegressor
"""
regressors:

 RandomForestRegressor
 SVR
 xgboost.XGBRegressor
 dc.models.XGBoostModel
 ligthgbm.LGBMRegressor
 catboost.CatBoostRegressor

"""
def train_model(regressor, X_train, y_train, label_scaler):
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1)
    regressor.fit(X_train, y_train)
    y_pred = label_scaler.inverse_transform(regressor.predict(X_validation))
    return mean_absolute_error(y_validation, y_pred)


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def finetune_xgb(X_train, y_train, X_test, label_scaler):
    # https://www.analyticsvidhya.com/blog/2021/06/predict-future-sales-using-xgbregressor/
    max_depth_values = [5, 9, 10, 14]
    min_child_weight_values = [1, 5, 6, 10]

    best_mae = 9999
    best_max_depth = None
    best_min_child_weight = None

    for max_depth in max_depth_values:
        for min_child_weight in min_child_weight_values:
            regressor = XGBRegressor(max_depth=max_depth, min_child_weight=min_child_weight)
            mae = train_model(regressor, X_train, y_train, label_scaler)
            print(best_mae, best_max_depth, best_min_child_weight)
            if mae < best_mae:
                best_mae = mae
                best_max_depth = max_depth
                best_min_child_weight = min_child_weight

    print("BEST MAX DEPTH: ********", best_max_depth)
    print("BEST MIN CHILD WEIGHT: ********", best_min_child_weight)

    subsample_values = [1, 0.8, 0.6, 0.3]
    cosample_bytree_values = [1, 0.8, 0.6, 0.3]

    best_mae = 999
    best_subsample = None
    best_cosample_bytree = None

    for subsample in subsample_values:
        for cosample_bytree in cosample_bytree_values:
            regressor = XGBRegressor(subsample=subsample, cosample_bytree=cosample_bytree)
            mae = train_model(regressor, X_train, y_train, label_scaler)
            print(best_mae, best_subsample, best_cosample_bytree)
            if mae < best_mae:
                best_mae = mae
                best_subsample = subsample
                best_cosample_bytree = cosample_bytree

    print("BEST SUBSAMPLE: ********", best_subsample)
    print("BEST COSAMPLE BYTREE: ********", best_cosample_bytree)

    alpha_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    best_mae = 999
    best_alpha = None
    for alpha in alpha_values:
        regressor = XGBRegressor(alpha=alpha)
        mae = train_model(regressor, X_train, y_train, label_scaler)
        print(best_mae, alpha)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    print("BEST ALPHA: ********", best_alpha)

"""
alpha=0.3, subsample=1, cosample_bytree=1, max_depth=10, min_child_weight=1
"""