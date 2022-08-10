# TODO fix
from data.TRAINING_csvs.training_splitter import load_for_explainability
from sklearn.feature_extraction.text import CountVectorizer
from alibi.explainers import AnchorTabular
from sklearn.svm import SVC


def predict_fn(model, datapoint):
    """

    :param model:
    :param datapoint:
    :return:
    """
    return model.predict(datapoint)


def main():
    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        data = load_for_explainability(label_column)
        cv = CountVectorizer(max_features=5000)
        X, y = data[text_column].to_list(), data[label_column].to_list()
        print(len(y))
        X = cv.fit_transform(X).toarray()
        model = SVC()
        print(model)

        feature_names = ["feat_" + str(i) for i in range(X.shape[-1])]

        # initialize and fit explainer by passing a prediction function and any other required arguments
        explainer = AnchorTabular(predict_fn, feature_names=feature_names, category_map=label_column)
        explainer.fit(X)

        # explain an instance
        explanation = explainer.explain(X)
        print(explanation)


if __name__ == "__main__":
    main()
