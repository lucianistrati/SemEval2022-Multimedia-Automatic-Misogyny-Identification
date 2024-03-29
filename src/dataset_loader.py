from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable

import random

import numpy as np
import pandas as pd

import pdb


def document_preprocess(document):
    """

    :param document:
    :return:
    """
    return document.lower().split()


def load_wce_dataset():
    """

    :return:
    """
    arr = np.load(file="data/wce_dataset.npy", allow_pickle=True)
    return [a for a in arr]


def load_abcnews_dataset():
    """

    :return:
    """
    arr = np.load(file="data/wce_dataset.npy", allow_pickle=True)
    df = pd.read_csv("data/abcnews-date-text.csv")
    return df['headline_text'].to_list() + [a for a in arr]


def main():
    texts = load_wce_dataset()
    # texts = load_abcnews_dataset()
    for text in texts[:5]:
        print(text)
    print(texts[-1])

    print(len(texts))
    train = []
    val = []
    test = []
    for i in range(len(texts)):
        text = texts[i]
        num = random.random()
        if num <= 0.8:
            train.append(text)
        elif 0.8 < num <= 0.9:
            val.append(text)
        elif 0.9 < num:
            test.append(text)

    separator = "\n"

    with open("corpus/train/train.txt", "w") as f:
        f.write(separator.join(train))

    with open("corpus/valid.txt", "w") as f:
        f.write(separator.join(val))

    with open("corpus/test.txt", "w") as f:
        f.write(separator.join(test))


if __name__ == '__main__':
    main()
