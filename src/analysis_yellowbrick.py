from yellowbrick.features import Rank2D

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.datasets import make_moons
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
    # print(X.shape, X.dtype, type(X))
    visualizer = Rank2D(features=features, algorithm='covariance')
    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.show() #poof()oof()                   # Draw/show/poof the data


    from sklearn.svm import LinearSVC, SVC
    from yellowbrick.classifier import ROCAUC
    model = LinearSVC()
    # model = SVC()
    model.fit(X, y)
    visualizer = ROCAUC(model)
    # visualizer.score(X,y)
    visualizer.show()


