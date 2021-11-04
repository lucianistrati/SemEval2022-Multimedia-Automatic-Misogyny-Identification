import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from process_data import load_train_data, load_test_data
import numpy as np
import pandas as pd
from typing import List

import random

def plot_wordcloud(text: str, label: int, classif_feature: str):
    title = f'{classif_feature}_texts_labelled_as_{label}'
    background_color = "black" if label == 1 else "white"
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color=background_color, stopwords=STOPWORDS).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig("data/data_analysis/text/wordclouds/" + title + ".png")
    plt.show()

def text_visualization(dataset_texts, dataset_labels, label_idxs, label_columns):
    for labels_idx in label_idxs:
        for label in [0, 1]:
            concat_text = ""
            for (text, labels_row) in list(zip(dataset_texts, dataset_labels)):
                if int(labels_row[labels_idx]) == label:
                    concat_text += (text + " ")
            # print(len(concat_text))
            plot_wordcloud(text=concat_text, label=label, classif_feature=label_columns[labels_idx])


def plot_images_stack(images: List, label: int, classif_feature: str):
    title = f'{classif_feature}_images_labelled_as_{label}'
    fig = plt.figure(figsize=(4, 4))
    columns = 4
    rows = 4
    for i in range(16):
        img = images[i]
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
    plt.savefig("data/data_analysis/images/image_stacks/" + title + ".png")
    plt.show()

def image_visualization(dataset_images, dataset_labels, label_idxs, label_columns):
    for labels_idx in label_idxs:
        for label in [0, 1]:
            selected_images = []
            for (image, labels_row) in list(zip(dataset_images, dataset_labels)):
                if int(labels_row[labels_idx]) == label:
                    selected_images.append(image)
            random.shuffle(selected_images)
            plot_images_stack(images=selected_images, label=label, classif_feature=label_columns[labels_idx])


def main():
    train_images, train_texts, train_labels = load_train_data("data/train_numpy_arrays")
    test_images, test_texts, test_labels = load_test_data("data/numpy_arrays")

    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
    label_idxs = list(range(5))

    text_visualization(train_texts, train_labels, label_idxs, label_columns)
    image_visualization(train_texts, train_labels, label_idxs, label_columns)

if __name__ == "__main__":
    main()