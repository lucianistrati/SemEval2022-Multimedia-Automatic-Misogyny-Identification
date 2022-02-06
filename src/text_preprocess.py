import textstat

textstat.set_lang("en")

import numpy as np
import torch
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasRegressor
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from typing import List

from src.feature_extractor import *

PAD_TOKEN = "__PAD__"
# word2vec_model = Word2Vec.load("src/embeddings_train/word2vec.model")

numpy_arrays_path = "data/numpy_data"
# word2vec_model = Word2Vec.load("src/embeddings_train/fasttext.model")

# from src.embeddings_train.train_word2vec import document_preprocess


# def document_preprocess(document):
#     return document.lower().split()

# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.syn1neg.npy")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.wv.vectors.npy")

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def embed_text(text, embedding_model: str = "sentence_transformer", dl_framework="tf"):
    """This function embedds the text given using an embedding model, it might also use the phrase, start_offset or the end_offset for certain embeddings"""
    # print(embedding_model)
    if embedding_model == "roberta":
        from transformers import RobertaTokenizer, RobertaModel
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaModel.from_pretrained('roberta-base')
        encoded_input = roberta_tokenizer(text, return_tensors=dl_framework)
        output = roberta_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        sent_transf_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embedding = sent_transf_model.encode(text)
        return embedding
    elif embedding_model == "roberta_large":
        from transformers import RobertaTokenizer, RobertaModel
        roberta_large_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        roberta_large_model = RobertaModel.from_pretrained('roberta-large')
        encoded_input = roberta_large_tokenizer(text, return_tensors=dl_framework)
        output = roberta_large_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "bert_large":
        from transformers import BertTokenizer, BertModel
        bert_large_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bert_large_model = BertModel.from_pretrained("bert-large-uncased")
        encoded_input = bert_large_tokenizer(text, return_tensors=dl_framework)
        output = bert_large_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "albert-base-v2":
        from transformers import AlbertTokenizer, AlbertModel
        albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        albert_model = AlbertModel.from_pretrained("albert-base-v2")
        encoded_input = albert_tokenizer(text, return_tensors=dl_framework)
        output = albert_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "roberta_distil":
        from sentence_transformers import SentenceTransformer
        sent_transf_model_distil_roberta = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v2")
        embedding = sent_transf_model_distil_roberta.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l12":
        from sentence_transformers import SentenceTransformer
        all_minimlm_l12_model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L12-cos-v5")
        embedding = all_minimlm_l12_model.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l6":
        from sentence_transformers import SentenceTransformer
        all_minimlm_l6_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        embedding = all_minimlm_l6_model.encode(text)
        return embedding
    elif embedding_model == "word2vec_trained":
        try:
            vector = word2vec_model.wv[document_preprocess(text)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector
    elif embedding_model == "paper_features":
        return get_paper_features(phrase, text, start_offset, end_offset)
    elif embedding_model == "word2vec_trained_special":
        try:
            vector = word2vec_model.wv[text]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector


from nltk.corpus import wordnet


def count_word_senses(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(len(wordnet.synsets(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_vowels(word):
    return len([c for c in word if c in "aeiou"])


def count_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    return len([c for c in word if c in consonants])


def count_double_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    cnt = 0
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i] in consonants and word[i + 1] in consonants:
            cnt += 1
    return cnt


def get_double_consonants_pct(word):
    return count_double_consonants(word) / len(word)


def get_vowel_pct(word):
    return count_vowels(word) / len(word)


def get_consonants_pct(word):
    return count_consonants(word) / len(word)


import nltk


def get_part_of_speech(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens == None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


from nltk.tokenize import TreebankWordTokenizer as twt


def spans(phrase):
    return list(twt().span_tokenize(phrase))


stop_words = set(stopwords.words('english'))


def count_sws(text, tokens=None):
    if tokens == None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


def get_context_tokens(phrase, start_offset, end_offset, context_size=1):  # try 2
    tokens = [PAD_TOKEN for _ in range(context_size)] + nltk.word_tokenize(phrase) + [PAD_TOKEN for _ in range(context_size)]
    tokens_spans = [(0, 0) for _ in range(context_size)] + spans(phrase) + [(0, 0) for _ in range(context_size)]
    for i, (l, r) in enumerate(tokens_spans):
        if r >= start_offset:
            return tokens[i - context_size: i] + tokens[i + 1: i + context_size + 1]
    return None


def embed_multiple_models(texts, submission_texts, labels, embedding_models: List[str], embedding_features: List[str], strategy: str = "averaging"):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    for embedding_model in zip(embedding_models):
        X_train = [embed_text(text=text, embedding_model=embedding_model) for text in texts]
        X_test = [embed_text(text=text, embedding_model=embedding_model) for text in submission_texts]
        X_train_list.append(X_train)
        y_train_list.append(labels)
        X_test_list.append(X_test)

    if strategy == "averaging":
        X_train_list = np.array(X_train_list)
        X_test_list = np.array(X_test_list)
        X_train = np.mean(X_train_list, axis=0)
        y_train = y_train_list[0]
        X_test = np.mean(X_test_list, axis=0)
    elif strategy == "stacking":
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    elif strategy == "ensemble":
        return X_train_list, y_train_list, X_test_list
    elif strategy == "dimensionality_reduction":
        reduction_method = "PCA"  # "LDA", "TSNE"
        if reduction_method == "PCA":
            dimensionality_reducer = PCA()
        elif reduction_method == "LDA":
            dimensionality_reducer = LDA()
        elif reduction_method == "TSNE":
            dimensionality_reducer = TSNE()
        for (X_train, X_test) in zip(X_train_list, X_test_list):
            X_train = dimensionality_reducer.fit_transform(X_train, n_components=min(X_train.shape))
            X_test = dimensionality_reducer.transform(X_test, n_components=min(X_test.shape))
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    return X_train, y_train, X_test


GOOD = 0
ERRORS = 0


def get_paper_features(phrase, target, start_offset, end_offset):
    context_tokens = get_context_tokens(phrase, start_offset, end_offset)
    if context_tokens == None:
        context_tokens = []
    word = target
    global ERRORS, GOOD
    target_ = target

    # so far 0.057701 just with target and the 24 features
    num_features = []

    for target in [target_]:  # + context_tokens:
        word = target
        num_features_ = [count_letters(target),
                         count_consonants(target),
                         count_vowels(target),
                         get_vowel_pct(target),
                         get_consonants_pct(target),
                         get_double_consonants_pct(target),
                         count_word_senses(target, tokens=word_tokenize(target)),
                         mean([count_word_senses(context_tok) for context_tok in context_tokens]),
                         get_base_word_pct(target, tokens=word_tokenize(word)),
                         has_suffix(target, tokens=word_tokenize(word)),
                         count_letters(target),
                         get_base_word_pct_stem(target, tokens=word_tokenize(word)),
                         has_both_affixes_stem(target, tokens=word_tokenize(word)),
                         count_hypernyms(target, tokens=word_tokenize(word)),
                         count_hyponyms(target, tokens=word_tokenize(word)),
                         count_antonyms(target, tokens=word_tokenize(word)),
                         count_definitions_average_tokens_length(target, tokens=word_tokenize(word)),
                         count_definitions_average_characters_length(target, tokens=word_tokenize(word)),
                         count_definitions_tokens_length(target, tokens=word_tokenize(word)),
                         count_total_phonemes_per_pronounciations(target, tokens=word_tokenize(word)),
                         get_word_frequency(target, tokens=word_tokenize(word)),
                         get_average_syllable_count(target),
                         check_word_compounding(target),
                         get_base_word_pct(target),
                         ]
        for feature in num_features_:
            num_features.append(feature)
        test_data = target
    """
    vectors = []
    for context_tok in context_tokens:
        if context_tok == PAD_TOKEN:
            continue
        try:
            # print(document_preprocess(context_tok))
            vector = word2vec_model.wv[document_preprocess(context_tok)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            GOOD += 1
        except KeyError:
             # continue
            vector = np.random.rand(1, 300)
            ERRORS += 1
        vectors.append(vector)

    import scipy
    try:
        # print(document_preprocess(context_tok))
        vector = word2vec_model.wv[document_preprocess(target)]
        vector = np.mean(vector, axis=0)
        vector = np.reshape(vector, (1, vector.shape[0]))
        GOOD += 1
    except KeyError:
        vector = np.random.rand(1, 300)
        ERRORS += 1

    target_vector = vector

    max_cos_sim = -1e18
    min_cos_sim = 1e18
    mean_cos_sim = 0.0

    for vector in vectors:
        ans = scipy.spatial.distance.cosine(np.reshape(vector, (vector.shape[-1], 1)), np.reshape(target_vector, (target_vector.shape[-1], 1)))
        max_cos_sim = max(max_cos_sim, ans)
        min_cos_sim = min(min_cos_sim, ans)
        mean_cos_sim += ans

    sum_cos_sim = copy.deepcopy(mean_cos_sim)
    mean_cos_sim /= len(vectors)
    """

    return num_features


if __name__ == '__main__':
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets('flexed'):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())
