from gensim.models import Word2Vec
from gensim.models import Word2Vec
# define training data
from gensim.test.utils import common_texts

from src.embeddings_train.dataset_loader import *

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt


def document_preprocess(document):
    return word_tokenize(document)


texts = load_wce_dataset()

# texts = load_abcnews_dataset()

texts = [document_preprocess(text) for text in texts]

import multiprocessing

print(multiprocessing.cpu_count())

import pdb

print(texts[0])
print(texts[-1])
model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count()) # an word2vec model is trained
model.save("word2vec.model") # checkpoint is saved
print("saved")
