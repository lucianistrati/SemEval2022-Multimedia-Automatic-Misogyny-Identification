from __future__ import print_function
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable

import pdb
import lime
import sklearn
import numpy as np
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from data.TRAINING_csvs.training_splitter import load_for_explainability

labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
text_column = "Text Transcription"

for label_column in labels_columns:
    data = load_for_explainability(label_column)
    df = data
    cv = CountVectorizer(max_features=5000)
    X, y = data[text_column].to_list(), data[label_column].to_list()
    X = cv.fit_transform(X).toarray()

    features = ["feat_" + str(i) for i in range(X.shape[-1])]

    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, newsgroups_train.target)

    pred = rf.predict(test_vectors)
    sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

    from lime import lime_text
    from sklearn.pipeline import make_pipeline

    c = make_pipeline(vectorizer, rf)

    print(c.predict_proba([newsgroups_test.data[0]]))

    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=class_names)

    idx = 0  # 83
    # TODO fix
    print(newsgroups_test.data[idx], c.predict_proba)
    exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    print('Document id: %d' % idx)
    print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0, 1])
    print('True class: %s' % class_names[newsgroups_test.target[idx]])

    exp.as_list()

    print('Original prediction:', rf.predict_proba(test_vectors[idx])[0, 1])
    tmp = test_vectors[idx].copy()
    tmp[0, vectorizer.vocabulary_['Posting']] = 0
    tmp[0, vectorizer.vocabulary_['Host']] = 0
    print('Prediction removing some features:', rf.predict_proba(tmp)[0, 1])
    print('Difference:', rf.predict_proba(tmp)[0, 1] - rf.predict_proba(test_vectors[idx])[0, 1])

    fig = exp.as_pyplot_figure()

    exp.show_in_notebook(text=False)

    exp.save_to_file('/tmp/oi.html')

    exp.show_in_notebook(text=True)
