import numpy as np
import os
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

numpy_arrays_path = "data/numpy_data"


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
