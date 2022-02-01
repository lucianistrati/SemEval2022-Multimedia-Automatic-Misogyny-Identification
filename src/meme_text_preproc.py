import contractions
import spacy
import unidecode
from bs4 import BeautifulSoup
from word2number import w2n

nlp = spacy.load('en_core_web_md')

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True,
                       convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True,
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True:  # remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True:  # remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True:  # remove accented characters
        text = remove_accented_chars(text)
    if contractions == True:  # expand contractions
        text = expand_contractions(text)
    if lowercase == True:  # convert all characters to lowercase
        text = text.lower()

    doc = nlp(text)  # tokenise text

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True:
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True:
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
                and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag == True:
            clean_text.append(edit)
    return clean_text


import re

from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = word_tokenize(text_rc)  # tokenization
    # print(tokens)
    # print(lemmatizer.lemmatize("worker"))
    # print(tokens)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]  # remove stopwords and stemming
    return " ".join(tokens)


from nltk.tokenize import word_tokenize


def clean_tweet(tweet):
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
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
                     if w.lower() in stopwords.words("english") or not w.isalpha())
    return tweet
