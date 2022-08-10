from interpret_text.common.utils_classical import get_important_words, BOWEncoder
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_text.unified_information import UnifiedInformationExplainer
from data.TRAINING_csvs.training_splitter import load_for_explainability
from sklearn.feature_extraction.text import CountVectorizer
from interpret_text.classical import ClassicalTextExplainer
from interpret_text.widget import ExplanationDashboard
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# TODO fix


def main():
    model = SVC()

    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        data = load_for_explainability(label_column)
        df = data
        print(df.columns)
        cv = CountVectorizer(max_features=5000)
        X, y = data[text_column].to_list(), data[label_column].to_list()
        print(len(y))
        X = cv.fit_transform(X).toarray()

        features = ["feat_" + str(i) for i in range(X.shape[-1])]
        train_dataset = None
        target_layer = None

        interpreter_unified = UnifiedInformationExplainer(model,
                                                          train_dataset,
                                                          target_layer)

        explainer = ClassicalTextExplainer()
        X_train = None
        y_train = None
        label_encoder = LabelEncoder()
        classifier, best_params = explainer.fit(X_train, y_train)

        print(features, interpreter_unified, label_encoder, best_params)

        HYPERPARAM_RANGE = {
            "solver": ["saga"],
            "multi_class": ["multinomial"],
            "C": [10 ** 4],
        }
        preprocessor = BOWEncoder()
        explainer = ClassicalTextExplainer(preprocessor, model, HYPERPARAM_RANGE)

        x_test = None
        # explain the first data point in the test set
        local_explanation = explainer.explain_local(x_test[0])

        # sorted feature importance values and feature names
        sorted_local_importance_names = local_explanation.get_ranked_local_names()
        sorted_local_importance_values = local_explanation.get_ranked_local_values()

        print(sorted_local_importance_values, sorted_local_importance_names)

        parsed_sentence_list = None
        feature_importance_values = None
        list_of_classes = [0, 1]
        name_of_model = None

        local_explanantion = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=feature_importance_values,
            method=name_of_model,
            model_task="classification",
            features=parsed_sentence_list,
            classes=list_of_classes,
        )

        print(ExplanationDashboard(local_explanantion))


if __name__ == "__main__":
    main()
