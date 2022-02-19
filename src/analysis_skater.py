import sklearn.svm
from sklearn.datasets import load_boston
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

    from skater import Interpretation
    interpreter = Interpretation(X, feature_names=features)

    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor()
    gb.fit(X, y)

    from skater.model import InMemoryModel
    model = InMemoryModel(gb.predict, examples = X[:10])
    import pandas as pd
    def predict_as_dataframe(x):
        return pd.DataFrame(gb.predict(x))

    from skater.model import InMemoryModel
    model = InMemoryModel(predict_as_dataframe, examples = X[:10])



    from skater.model import DeployedModel
    import numpy as np

    def input_formatter(x): return {'data': list(x)}
    def output_formatter(response): return np.array(response.json()['output'])
    # uri = "https://yourorg.com/model/endpoint"
    # model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10])

    from skater.model import DeployedModel
    import numpy as np

    # req_kwargs = {'cookies': {'cookie-name':'cookie'}}
    # model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10], request_kwargs=req_kwargs)

    def numpy_to_json(numpy_array):
       return [{'data':x} for x in numpy_array]

    skater_model = InMemoryModel(model.predict, input_formatter = numpy_to_json)

    unique_classes = [0, 1]
    classifier = sklearn.svm.SVC()
    skater_model = InMemoryModel(classifier.predict)#, unique_classes=unique_classes)


    # TODO fix
    # interpreter.feature_importance.feature_importance(skater_model)
    #
    # interpreter.partial_dependence.plot_partial_dependence([features[0], features[1]], skater_model)
    #
    # from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
    # LimeTabularExplainer(regressor_X, feature_names=regressor_data.feature_names, mode="regression").explain_instance(regressor_X[0], annotated_model)
    #
    # from skater.core.global_interpretation.interpretable_models.brlc import BRLC
    # sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)
