import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
import numpy as np
import spacy
import unidecode
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from word2number import w2n
import nltk
import emoji
import string


def load_lemmatizer():
    """

    :return:
    """
    return WordNetLemmatizer()


def load_nlp():
    """

    :return:
    """
    return spacy.load('en_core_web_md')


def deselect_stop_words():
    """

    :return:
    """
    nlp = load_nlp()
    # exclude words from spacy stopwords list
    deselect_stop_words = ['no', 'not']
    for w in deselect_stop_words:
        nlp.vocab[w].is_stop = False
    return nlp


def strip_html_tags(text):
    """
    remove html tags from text
    :param text:
    :return:
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """
    remove extra whitespaces from text
    :param text:
    :return:
    """
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """
    remove accented characters from text, e.g. café
    :param text:
    :return:
    """
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """
    expand shortened words, e.g. don't to do not
    :param text:
    :return:
    """
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True,
                       convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True,
                       stop_words=True):
    """
    preprocess text with default option set to true for all steps
    :param text:
    :param accented_chars:
    :param contractions:
    :param convert_num:
    :param extra_whitespace:
    :param lemmatization:
    :param lowercase:
    :param punctuations:
    :param remove_html:
    :param remove_num:
    :param special_chars:
    :param stop_words:
    :return:
    """
    if remove_html is True:  # remove html tags
        text = strip_html_tags(text)
    if extra_whitespace is True:  # remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars is True:  # remove accented characters
        text = remove_accented_chars(text)
    if contractions is True:  # expand contractions
        text = expand_contractions(text)
    if lowercase is True:  # convert all characters to lowercase
        text = text.lower()

    nlp = load_nlp()
    doc = nlp(text)  # tokenise text

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words is True and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations is True and token.pos_ == 'PUNCT' and flag is True:
            flag = False
        # remove special characters
        if special_chars is True and token.pos_ == 'SYM' and flag is True:
            flag = False
        # remove numbers
        if remove_num is True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
                and flag is True:
            flag = False
        # convert number words to numeric numbers
        if convert_num is True and token.pos_ == 'NUM' and flag is True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization is True and token.lemma_ != "-PRON-" and flag is True:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag is True:
            clean_text.append(edit)
    return clean_text


def clean_text(text, lemmatizer=None):
    """

    :param text:
    :param lemmatizer:
    :return:
    """
    if lemmatizer is None:
        lemmatizer = load_lemmatizer()
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = word_tokenize(text_rc)  # tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)


def clean_tweet(tweet):
    """

    :param tweet:
    :return:
    """
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp)  # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("#[A-Za-z0-9_]+", "", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]', ' ', temp)
    temp = re.sub("[^a-z0-9]", " ", temp)
    temp = word_tokenize(temp)
    # temp = [lemmatizer.lemmatize(w) for w in temp]
    temp = " ".join(word for word in temp)
    return temp


def cleaner(tweet):
    """

    :param tweet:
    :return:
    """
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
                     if w.lower() in stopwords.words("english") or not w.isalpha())
    return tweet


def main():
    pass


if __name__ == "__main__":
    main()
