# from shifterator import relative_shift as rs
#
# # Get a sentiment word shift
# sentiment_shift = rs.sentiment_shift(reference=word2freq_ref,
#                                      comparison=word2freq_comp
# sent_dict_ref = word2score_ref,
#                 sent_dict_comp = word2score_comp)
# sentiment_shift.get_shift_graph()
#
# # Get an entropy shift
# entropy_shift = rs.entropy_shift(reference=type2freq_ref,
#                                  comparison=type2freq_comp,
#                                  base=2)
# entropy_shift.get_shift_graph()
#
# # Get a Kullback-Leibler divergence shift
# # Note: only well-defined if both texts have all the same words
# kld_shift = rs.kl_divergence_shift(reference=word2freq_ref,
#                                    comparison=word2freq_comp,
#                                    base=2)
# kld_shift.get_shift_graph()
#
# # Get a Jensen-Shannon divergence shift
# from shifterator import symmetric_shift as ss
#
# jsd_shift = ss.js_divergence_shift(system_1=word2freq_1,
#                                    system_2=word2freq_2,
#                                    base=2)
# jsd_shift.get_shift_graph()
#
# from shifterator import shifterator as sh
#
# # Construct a general shift object
# shift = sh.Shift(system_1=type2freq_1,
#                  system_2=type2freq_2,
#                  type2score_1=type2score_1,
#                  type2score_2=type2score_2,
#                  reference_val=None,
#                  stop_lens=None)
#
# # Get a weighted average using a Shift object
# shift = sh.Shift()
# weighted_avg = shift.get_weighted_score(word2freq, word2score)
#
# # Get shift scores of each word as a dictionary
# type2shift_scores = shift.get_shift_scores(details=False)
#
# # Get the components of the shift score for each word
# type2p_avg, type2s_diff, type2p_diff, type2s_ref_diff, type2shift_scores = shift.get_shift_scores()
#
# # Get the total sum of each type of contribution
# # Order: Positive freq and positive score, negative freq and positive score,
# #        Positive freq and negative score, negative freq and negative score,
# #        Positive score diff, negative score diff
# shift_components = shift.get_shift_component_sums()
#
# # Set a stop lens on a Shift object
# sentiment_shift = rs.sentiment_shift(reference=word2freq1,
#                                      comparison=word2freq2,
#                                      type2score=word2score,
#                                      stop_lens=[(4, 6), (0, 1), (8, 9)])
#
# # Manually set reference value on a Shift object
# jsd_shift = ss.js_divergence_shift(system_1=word2freq_1,
#                                    system_2=word2freq_2,
#                                    reference_value=0)
#
# """
#
# Plotting Parameters
#
# There are a number of plotting parameters that can be passed to get_shift_graph()
# when constructing a word shift graph. See get_plotting_params()
# for the parameters that can currently altered in a word shift graph.
# """



###################################

# Import packages

import pandas as pd
import numpy as np
import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re

# from shifterator import relative_shift as rs
from shifterator import shifts as rs

import matplotlib.pyplot as plt
import seaborn as sns

from data.TRAINING_csvs.training_splitter import load_for_explainability

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


    texts = reviews[text_column].tolist()
    texts_neg = reviews_neg[text_column].tolist()
    texts_pos = reviews_pos[text_column].tolist()

    # We will want to remove stop words
    stop_words = set(stopwords.words('english'))


    from collections import Counter
    def clean_text(texts):
        return Counter(texts)

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



    # From https://www.kaggle.com/prakashsadashivappa/word-cloud-of-abstracts-cord-19-dataset
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black',
        ).generate(str(texts_neg))



    fig = plt.figure(
        figsize = (10, 8),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()





    # Get an entropy shift
    entropy_shift = rs.EntropyShift(type2freq_1=clean_texts_neg, type2freq_2=clean_texts_pos, base=2)
    entropy_shift.get_shift_graph()



    # Get a Jensen-Shannon divergence shift
    from shifterator import shifts as ss
    jsd_shift = ss.JSDivergenceShift(type2freq_1=clean_texts_neg,
            type2freq_2=clean_texts_pos,
                                     base=2)
    jsd_shift.get_shift_graph()
