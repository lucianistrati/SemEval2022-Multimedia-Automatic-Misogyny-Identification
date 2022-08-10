import sklearn.svm
from sklearn.datasets import load_boston
from sklearn.feature_extraction.text import CountVectorizer

from data.TRAINING_csvs.training_splitter import load_for_explainability
from sklearn.ensemble import GradientBoostingRegressor
from skater.model import DeployedModel
from skater.model import InMemoryModel
from skater import Interpretation

import pandas as pd
import numpy as np


def input_formatter(x):
    """

    :param x:
    :return:
    """
    return {'data': list(x)}


def predict_as_dataframe(x, gb=None):
    """

    :param x:
    :param gb:
    :return:
    """
    if gb is None:
        gb = GradientBoostingRegressor()
    return pd.DataFrame(gb.predict(x))


def output_formatter(response):
    """

    :param response:
    :return:
    """
    return np.array(response.json()['output'])


def numpy_to_json(numpy_array):
    """

    :param numpy_array:
    :return:
    """
    return [{'data': x} for x in numpy_array]


def main():
    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        data = load_for_explainability(label_column)
        cv = CountVectorizer(max_features=5000)
        X, y = data[text_column].to_list(), data[label_column].to_list()
        X = cv.fit_transform(X).toarray()

        gb = GradientBoostingRegressor()
        gb.fit(X, y)

        model = InMemoryModel(predict_as_dataframe, examples=X[:10])
        skater_model = InMemoryModel(model.predict, input_formatter=numpy_to_json)
        print(skater_model)
        classifier = sklearn.svm.SVC()
        skater_model = InMemoryModel(classifier.predict)

        print(skater_model)


if __name__ == "__main__":
    main()
