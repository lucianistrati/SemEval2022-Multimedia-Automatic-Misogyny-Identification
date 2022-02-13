from interpret_text.unified_information import UnifiedInformationExplainer

interpreter_unified = UnifiedInformationExplainer(model,
                                 train_dataset,
                                 device,
                                 target_layer)

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
