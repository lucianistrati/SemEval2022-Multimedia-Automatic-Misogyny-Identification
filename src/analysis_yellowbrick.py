from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from yellowbrick.features import Rank2D

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.datasets import make_moons
from data.TRAINING_csvs.training_splitter import load_for_explainability
from sklearn.svm import LinearSVC, SVC
from yellowbrick.classifier import ROCAUC

import pdb


def main():
    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        data = load_for_explainability(label_column)
        df = data
        print(df.head())
        cv = CountVectorizer(max_features=5000)
        X, y = data[text_column].to_list(), data[label_column].to_list()
        X = cv.fit_transform(X).toarray()
        features = ["feat_" + str(i) for i in range(X.shape[-1])]
        visualizer = Rank2D(features=features, algorithm='covariance')
        visualizer.fit(X, y)
        visualizer.transform(X)
        visualizer.show()

        model = LinearSVC()
        # model = SVC()
        model.fit(X, y)
        visualizer = ROCAUC(model)
        # visualizer.score(X,y)
        visualizer.show()


if __name__ == "__main__":
    main()
