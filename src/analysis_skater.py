from sklearn.datasets import load_boston
boston = load_boston()
X, y, features = boston.data, boston.target, boston.feature_names

from skater import Interpretation
interpreter = Interpretation(X, feature_names=features)

from sklearn.ensemble import GradientBoostedRegressor
gb = GradientBoostedRegressor()
gb.fit(X, y)

from skater.model import InMemoryModel
model = InMemoryModel(gb.predict, examples = X[:10])

def predict_as_dataframe(x):
    return pd.DataFrame(gb.predict(x))

from skater.model import InMemoryModel
model = InMemoryModel(predict_as_dataframe, examples = X[:10])



from skater.model import DeployedModel
import numpy as np

def input_formatter(x): return {'data': list(x)}
def output_formatter(response): return np.array(response.json()['output'])
uri = "https://yourorg.com/model/endpoint"
model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10])

from skater.model import DeployedModel
import numpy as np

req_kwargs = {'cookies': {'cookie-name':'cookie'}}
model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10], request_kwargs=req_kwargs)

def numpy_to_json(numpy_array):
   return [{'data':x} for x in numpy_array]

skater_model = InMemoryModel(model.predict, input_formatter = numpy_to_json)

unique_classes = [0, 1]
skater_model = InMemoryModel(classifier.predict, unique_classes=unique_classes)


interpreter.feature_importance.feature_importance(skater_model)

interpreter.partial_dependence.plot_partial_dependence([features[0], features[1]], skater_model)

from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
LimeTabularExplainer(regressor_X, feature_names=regressor_data.feature_names,
mode="regression").explain_instance(regressor_X[0], annotated_model)

from skater.core.global_interpretation.interpretable_models.brlc import BRLC
sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)

