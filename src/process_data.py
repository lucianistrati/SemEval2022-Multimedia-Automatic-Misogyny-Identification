import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_data_folder(folder_path: str, data_set: str, first_character: str):
    """

    :param folder_path:
    :param data_set:
    :param first_character:
    :return:
    """
    images_dict = dict()
    images = []
    texts = []
    labels = []
    preview = False
    for (dirpath, _, filenames) in os.walk(folder_path):
        for filename in tqdm(filenames):
            if filename.endswith('.jpg') and filename[1] != '.' and int(filename[0:2]) != first_character:
                continue
            if filename.endswith('.jpg'):
                image_path = os.path.join(dirpath, filename)
                image = cv2.imread(image_path)
                # print(type(image))
                images_dict[int(filename[:-4])] = image
                if preview:
                    cv2.imshow('image', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            elif filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, filename), delimiter='\t', error_bad_lines=False)

    print(df.head())
    print(df.columns)
    print(len(df))

    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]

    for i in range(len(df)):
        if df.at[i, "file_name"][1] != '.' and int(df.at[i, 'file_name'][:2]) != first_character:
            continue
        row_labels = []
        for column in list(df.columns):
            if column in label_columns:
                row_labels.append(df.at[i, column])
        labels.append(row_labels)
        images.append(images_dict[int(df.at[i, "file_name"][:-4])])
        texts.append(df.at[i, "Text Transcription"])

    images = np.array(images)
    texts = np.array(texts)
    labels = np.array(labels)

    print(images.shape)
    print(texts.shape)
    print(labels.shape)

    # np.save(folder_path + "/" + data_set + "_" + str(first_character) + "_images.npy", images, allow_pickle=True)
    # np.save(folder_path + "/" + data_set + "_" + str(first_character) + "_texts.npy", texts, allow_pickle=True)
    # np.save(folder_path + "/" + data_set + "_" + str(first_character) + "_labels.npy", labels, allow_pickle=True)


def merge_train_npy_arrays(folder_path: str) -> None:
    """

    :param folder_path:
    :return:
    """
    images = []
    texts = []
    labels = []
    for i in tqdm(range(2, 20)):
        sub_images = np.load(folder_path + "/train_" + str(i) + "_images.npy", allow_pickle=True)
        sub_texts = np.load(folder_path + "/train_" + str(i) + "_texts.npy", allow_pickle=True)
        sub_labels = np.load(folder_path + "/train_" + str(i) + "_labels.npy", allow_pickle=True)
        for image in sub_images:
            images.append(image)
        for text in sub_texts:
            texts.append(text)
        for label in sub_labels:
            labels.append(label)

    images = np.array(images)
    texts = np.array(texts)
    labels = np.array(labels)

    print(images.shape)
    print(texts.shape)
    print(labels.shape)

    # np.save("data/numpy_arrays/train_images.npy", images, allow_pickle=True)
    # np.save("data/numpy_arrays/train_texts.npy", texts, allow_pickle=True)
    # np.save("data/numpy_arrays/train_labels.npy", labels, allow_pickle=True)


def load_train_data(folder_path: str):
    """

    :param folder_path:
    :return:
    """
    train_images = np.load(os.path.join(folder_path, "train_10_images.npy"), allow_pickle=True),
    train_texts = np.load(os.path.join(folder_path, "train_10_texts.npy"), allow_pickle=True)
    train_labels = np.load(os.path.join(folder_path, "train_10_labels.npy"), allow_pickle=True)
    return train_images, train_texts, train_labels


def load_test_data(folder_path: str):
    """

    :param folder_path:
    :return:
    """
    test_images = np.load(os.path.join(folder_path, "test_images.npy"), allow_pickle=True)
    test_texts = np.load(os.path.join(folder_path, "test_texts.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(folder_path, "test_labels.npy"), allow_pickle=True)
    return test_images, test_texts, test_labels


def main():
    train_path = "data/TRAINING"
    test_path = "data/trial"
    merge_train_npy_arrays("data/train_numpy_arrays")
    for digit in range(10, 20):
        parse_data_folder(train_path, "train", digit)
    parse_data_folder(test_path, "test", 10)
    train_images, train_texts, train_labels = load_train_data("data/train_numpy_arrays")
    test_images, test_texts, test_labels = load_test_data("data/numpy_arrays")

    print(len(train_images), len(test_images))


if __name__ == '__main__':
    main()
