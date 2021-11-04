import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from process_data import load_train_data, load_test_data
import numpy as np
import pandas as pd
from typing import List

import random
import cv2


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
        # if i <= 3:
        #     plt.title(title)
    plt.savefig("data/data_analysis/images/image_stacks/" + title + ".png")
    plt.title(title)
    # plt.show()

def image_visualization(dataset_images, dataset_labels, label_idxs, label_columns):
    for labels_idx in label_idxs:
        for label in [0, 1]:
            selected_images = []
            for (image, labels_row) in list(zip(dataset_images, dataset_labels)):
                if int(labels_row[labels_idx]) == label:
                    selected_images.append(image)
            random.shuffle(selected_images)
            print(len(selected_images))
            plot_images_stack(images=selected_images[:16], label=label, classif_feature=label_columns[labels_idx])

def dataframe_visualization(df):
    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]

    for label_column in label_columns:
        labels = df[label_column].to_list()
        print(label_column, f'0: {len([label for label in labels if label == 0])}, '
                            f'1:{len([label for label in labels if label == 1])}')
        labels_proportion = [len([label for label in labels if label == 0]),
                       len([label for label in labels if label == 1])]
        print(labels_proportion)
        np.save(arr=np.array(labels_proportion),
                file="data/labels_proportions/labels_proportions_" + label_column + ".npy", allow_pickle=True)


def main():
    train_images, train_texts, train_labels = load_train_data("data/train_numpy_arrays")
    test_images, test_texts, test_labels = load_test_data("data/numpy_arrays")
    df = pd.read_csv("data/TRAINING/training.csv", delimiter='\t', error_bad_lines=False)
    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
    label_idxs = list(range(5))

    # text_visualization(train_texts, train_labels, label_idxs, label_columns)
    # image_visualization(train_images, train_labels, label_idxs, label_columns)

    dataframe_visualization(df)

if __name__ == "__main__":
    main()