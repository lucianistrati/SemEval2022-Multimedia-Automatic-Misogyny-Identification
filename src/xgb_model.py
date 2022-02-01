import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import BayesianRidge, HuberRegressor

from sklearn.model_selection import KFold


def train_basic_model(X_train, y_train, X_test, embedding_feature: str = "target_word", embedding_model: str = "roberta"):
    data_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # regressor = XGBRegressor(eval_metric=mean_absolute_error,  max_depth=10, min_child_weight=1) #alpha=0.3, subsample=1, cosample_bytree=1,)
    # regressor = SVR() #
    # regressor = DecisionTreeRegressor() #
    # regressor = AdaBoostRegressor() #
    # regressor = GradientBoostingRegressor() #
    # regressor = HuberRegressor() #
    # regressor = SGDRegressor() #
    # regressor = LinearRegression() #
    # regressor = MLPRegressor() #
    # regressor = BayesianRidge()
    # regressor = RandomForestRegressor(random_state=100)  #

    def cross_val_func(regressor, X_train, y_train):
        kfolder5 = KFold(n_splits=5, shuffle=False)
        print(X_train.shape)
        scores = cross_val_score(regressor, X_train, y_train, scoring='neg_mean_absolute_error', cv=kfolder5, n_jobs=-1)
        return [score * (-1) for score in scores]

    regressors_list = [XGBRegressor(), SVR(), DecisionTreeRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), HuberRegressor(),
                       SGDRegressor(), MLPRegressor(), BayesianRidge(), RandomForestRegressor()]
    regressor_names = ["random forest", "linear regressor", "svr"]
    for (regressor, name) in list(zip(regressors_list, regressor_names)):
        print(name, ":", cross_val_func(regressor, X_train, y_train))
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
