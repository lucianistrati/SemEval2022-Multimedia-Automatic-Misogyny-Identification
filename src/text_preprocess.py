from used_repos.personal.aggregated_personal_repos.semeval.src.feature_extractor import check_word_compounding, \
    count_antonyms, count_average_phonemes_per_pronounciation, count_capital_chars, count_capital_words, \
    count_definitions_average_characters_length, count_definitions_average_tokens_length, \
    count_definitions_characters_length, count_definitions_tokens_length, count_entailments, count_holonyms, \
    count_hypernyms, count_hyponyms, count_letters, count_meronyms, count_part_holonyms, count_part_meroynms, \
    count_pronounciation_methods, count_punctuations, count_substance_holonyms, count_substance_meroynms, \
    count_synonyms, count_total_phonemes_per_pronounciations, count_troponyms, custom_wup_similarity, \
    get_average_syllable_count, get_base_word_pct, get_base_word_pct_stem, get_num_pos_tags, get_phrase_len, \
    get_phrase_num_tokens, get_target_phrase_ratio, get_total_syllable_count, get_word_frequency, \
    get_word_position_in_phrase, get_wup_avg_similarity, has_both_affixes, has_both_affixes_stem, has_prefix, \
    has_prefix_stem, has_suffix, has_suffix_stem, is_plural, is_singular, mean, median, word_frequency, \
    word_origin, word_polarity, word_tokenize
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from tensorflow.python.ops.numpy_ops import np_config
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.corpus import wordnet
from nltk import wordnet as wn

import numpy as np

import textstat
import torch
import nltk
import pdb


# from keras.layers import Dense
# from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasRegressor
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from gensim.models import Word2Vec


def document_preprocess(text: str):
    """

    :param text:
    :return:
    """
    return word_tokenize(text)


textstat.set_lang("en")


def load_word2vec_model():
    """

    :return:
    """
    return Word2Vec.load("src/embeddings_train/word2vec.model")


PAD_TOKEN = "__PAD__"
np_config.enable_numpy_behavior()
numpy_arrays_path = "data/numpy_data"


def embed_text(text, embedding_model: str = "sentence_transformer", dl_framework="tf"):
    """
    This function embedds the text given using an embedding model, it might also use the phrase, start_offset or
    the end_offset for certain embeddings
    :param text:
    :param embedding_model:
    :param dl_framework:
    :return:
    """
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
            word2vec_model = load_word2vec_model()
            vector = word2vec_model.wv[document_preprocess(text)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector
    elif embedding_model == "paper_features":
        phrase = ""
        start_offset = 0
        end_offset = -1
        return get_paper_features(phrase, text, start_offset, end_offset)
    elif embedding_model == "word2vec_trained_special":
        try:
            phrase = ""
            start_offset = 0
            end_offset = -1
            word2vec_model = load_word2vec_model()
            vector = word2vec_model.wv[text]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector


def count_word_senses(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(len(wordnet.synsets(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_vowels(word):
    """

    :param word:
    :return:
    """
    return len([c for c in word if c in "aeiou"])


def count_consonants(word):
    """

    :param word:
    :return:
    """
    consonants = "bcdfghjklmnpqrstvwxyz"
    return len([c for c in word if c in consonants])


def count_double_consonants(word):
    """

    :param word:
    :return:
    """
    consonants = "bcdfghjklmnpqrstvwxyz"
    cnt = 0
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i] in consonants and word[i + 1] in consonants:
            cnt += 1
    return cnt


def get_double_consonants_pct(word):
    """

    :param word:
    :return:
    """
    return count_double_consonants(word) / len(word)


def get_vowel_pct(word):
    """

    :param word:
    :return:
    """
    return count_vowels(word) / len(word)


def get_consonants_pct(word):
    """

    :param word:
    :return:
    """
    return count_consonants(word) / len(word)


def get_part_of_speech(sentence, tokens=None):
    """

    :param sentence:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(sentence) if tokens is None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    """

    :return:
    """
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


def spans(phrase):
    """

    :param phrase:
    :return:
    """
    return list(twt().span_tokenize(phrase))


stop_words = set(stopwords.words('english'))


def count_sws(text, tokens=None):
    """

    :param text:
    :param tokens:
    :return:
    """
    if tokens is None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    """

    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


def get_context_tokens(phrase, start_offset, end_offset, context_size=1):  # try 2
    """

    :param phrase:
    :param start_offset:
    :param end_offset:
    :param context_size:
    :return:
    """
    tokens = [PAD_TOKEN for _ in range(context_size)] + nltk.word_tokenize(phrase) + [PAD_TOKEN for _ in range(context_size)]
    tokens_spans = [(0, 0) for _ in range(context_size)] + spans(phrase) + [(0, 0) for _ in range(context_size)]
    for i, (l, r) in enumerate(tokens_spans):
        if r >= start_offset:
            return tokens[i - context_size: i] + tokens[i + 1: i + context_size + 1]
    return None


def embed_multiple_models(texts, submission_texts, labels, embedding_models: List[str], embedding_features: List[str],
                          strategy: str = "averaging"):
    """

    :param texts:
    :param submission_texts:
    :param labels:
    :param embedding_models:
    :param embedding_features:
    :param strategy:
    :return:
    """
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
    """

    :param phrase:
    :param target:
    :param start_offset:
    :param end_offset:
    :return:
    """
    context_tokens = get_context_tokens(phrase, start_offset, end_offset)
    if context_tokens is None:
        context_tokens = []
    global ERRORS, GOOD
    target_ = target

    # so far 0.057701 just with target and the 24 features
    num_features = []

    for target in [target_]:
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
        # test_data = target

    return num_features


def main():
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets('flexed'):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())


if __name__ == "__main__":
    main()
