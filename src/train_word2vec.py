from dataset_loader import load_wce_dataset, load_abcnews_dataset
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import multiprocessing


def document_preprocess(document):
    """

    :param document:
    :return:
    """
    return word_tokenize(document)


def main():
    texts = load_abcnews_dataset()

    texts = [document_preprocess(text) for text in texts]

    print(multiprocessing.cpu_count())

    print(texts[0])
    print(texts[-1])
    model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec.model")
    print("saved")


if __name__ == "__main__":
    main()
