from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# define training data

stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def document_preprocess(document):
    return word_tokenize(document)


# texts = load_abcnews_dataset()

texts = [document_preprocess(text) for text in texts]

import multiprocessing

print(multiprocessing.cpu_count())

print(texts[0])
print(texts[-1])
model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count())  # an word2vec model is trained
model.save("word2vec.model")  # checkpoint is saved
print("saved")
