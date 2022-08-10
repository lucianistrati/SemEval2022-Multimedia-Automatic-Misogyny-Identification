from data.TRAINING_csvs.training_splitter import load_for_explainability
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from shifterator import shifts as ss
from nltk.corpus import stopwords
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import collections
import itertools
import nltk
import re


# From https://www.kaggle.com/prakashsadashivappa/word-cloud-of-abstracts-cord-19-dataset


def clean_text(texts):
    """

    :param texts:
    :return:
    """
    return Counter(texts)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")

    # Load the review CSV

    labels_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    text_column = "Text Transcription"

    for label_column in labels_columns:
        reviews = load_for_explainability(label_column)
        reviews.head()

        fig, ax = plt.subplots(figsize=(12, 8))

        reviews_neg = reviews[reviews[label_column] == 0]
        reviews_pos = reviews[reviews[label_column] == 1]

        texts_neg = reviews_neg[text_column].tolist()
        texts_pos = reviews_pos[text_column].tolist()

        # Clean up the review texts
        clean_texts_neg = clean_text(texts_neg)
        clean_texts_pos = clean_text(texts_pos)

        # Dataframes for most frequent common words in positive and negative reviews
        common_neg = pd.DataFrame(clean_texts_neg.most_common(15),
                                  columns=['words', 'count'])
        common_pos = pd.DataFrame(clean_texts_pos.most_common(15),
                                  columns=['words', 'count'])

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot horizontal bar graph
        common_neg.sort_values(by='count').plot.barh(x='words',
                                                     y='count',
                                                     ax=ax,
                                                     color="red")

        ax.set_title("Common Words Found in Negative Reviews")

        plt.show()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot horizontal bar graph
        common_pos.sort_values(by='count').plot.barh(x='words',
                                                     y='count',
                                                     ax=ax,
                                                     color="green")

        ax.set_title("Common Words Found in Positive Reviews")

        plt.show()

        wordcloud = WordCloud(
            width=3000,
            height=2000,
            background_color='black',
        ).generate(str(texts_neg))

        plt.figure(
            figsize=(10, 8),
            facecolor='k',
            edgecolor='k')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

        # Get an entropy shift
        entropy_shift = ss.EntropyShift(type2freq_1=clean_texts_neg, type2freq_2=clean_texts_pos, base=2)
        entropy_shift.get_shift_graph()

        # Get a Jensen-Shannon divergence shift

        jsd_shift = ss.JSDivergenceShift(type2freq_1=clean_texts_neg,
                                         type2freq_2=clean_texts_pos,
                                         base=2)
        jsd_shift.get_shift_graph()


if __name__ == "__main__":
    main()
