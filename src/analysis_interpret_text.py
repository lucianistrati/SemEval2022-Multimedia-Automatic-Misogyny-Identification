from interpret_text.unified_information import UnifiedInformationExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
model = SVC()
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


    interpreter_unified = UnifiedInformationExplainer(model,
                                     train_dataset,
                                     target_layer)
    # TODO fix
    from sklearn.preprocessing import LabelEncoder
    from interpret_text.classical import ClassicalTextExplainer

    explainer = ClassicalTextExplainer()
    label_encoder = LabelEncoder()
    classifier, best_params = explainer.fit(X_train, y_train)

    from sklearn.preprocessing import LabelEncoder
    from interpret_text.classical import ClassicalTextExplainer
    from interpret_text.common.utils_classical import get_important_words, BOWEncoder

    HYPERPARAM_RANGE = {
        "solver": ["saga"],
        "multi_class": ["multinomial"],
        "C": [10 ** 4],
    }
    preprocessor = BOWEncoder()
    explainer = ClassicalTextExplainer(preprocessor, model, HYPERPARAM_RANGE)

    # explain the first data point in the test set
    local_explanation = explainer.explain_local(x_test[0])

    # sorted feature importance values and feature names
    sorted_local_importance_names = local_explanation.get_ranked_local_names()
    sorted_local_importance_values = local_explanation.get_ranked_local_values()

    from interpret_text.widget import ExplanationDashboard


    ExplanationDashboard(local_explanantion)

    from interpret_text.explanation.explanation import _create_local_explanation

    local_explanantion = _create_local_explanation(
    classification=True,
    text_explanation=True,
    local_importance_values=feature_importance_values,
    method=name_of_model,
    model_task="classification",
    features=parsed_sentence_list,
    classes=list_of_classes,
    )
