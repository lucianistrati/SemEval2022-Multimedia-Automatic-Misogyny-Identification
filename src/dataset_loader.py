import pandas as pd
import numpy as np
import random
def document_preprocess(document):
	return document.lower().split()

def load_wce_dataset():
    arr = np.load(file="data/wce_dataset.npy", allow_pickle=True)
    return [a for a in arr]

def load_abcnews_dataset():
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

    with open("corpus/train/train.txt", "w") as f:
        f.write("\n".join(train))

    with open("corpus/valid.txt", "w") as f:
        f.write("\n".join(val))

    with open("corpus/test.txt", "w") as f:
        f.write("\n".join(test))


if __name__=='__main__':
    main()