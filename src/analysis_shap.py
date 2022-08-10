from data.TRAINING_csvs.training_splitter import load_for_explainability
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

import keras.backend as K
import numpy as np

import transformers
import xgboost
import sklearn
import shap
import json


def main():
    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        data = load_for_explainability(label_column)
        cv = CountVectorizer(max_features=5000)
        X, y = data[text_column].to_list(), data[label_column].to_list()
        X = cv.fit_transform(X).toarray()

        model = xgboost.XGBRegressor().fit(X, y)

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # visualize the first prediction's explanation
        shap.plots.waterfall(shap_values[0])

        # visualize the first prediction's explanation with a force plot
        shap.plots.force(shap_values[0])

        # visualize all the training set predictions
        shap.plots.force(shap_values)

        # create a dependence scatter plot to show the effect of a single feature across the whole dataset
        shap.plots.scatter(shap_values[:, "RM"], color=shap_values)

        # summarize the effects of all the features
        shap.plots.beeswarm(shap_values)

        shap.plots.bar(shap_values)

        shap.initjs()

        # train a SVM classifier
        X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
        svm = sklearn.svm.SVC(kernel='rbf', probability=True)
        svm.fit(X_train, Y_train)

        # use Kernel SHAP to explain test set predictions
        explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
        shap_values = explainer.shap_values(X_test, nsamples=100)

        # plot the SHAP values for the Setosa output of the first instance
        shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :], link="logit")

        # plot the SHAP values for the Setosa output of all instances
        shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")


if __name__ == "__main__":
    main()
