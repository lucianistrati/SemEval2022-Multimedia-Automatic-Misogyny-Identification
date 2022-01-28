from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from statistics import mean

POSSIBLE_LABELS = [0, 1]

def get_sample_weights(y_train):
    from sklearn.utils import class_weight
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    return classes_weights


def task_a(texts, labels, model, submit=False, submission_texts=None, file_paths=None):
    if submit:
        model = XGBClassifier()
        model.fit(texts, labels, sample_weight=get_sample_weights(labels))
        y_pred = model.predict(submission_texts)
        submission_list = list(zip(file_paths, y_pred))
        submission_list = sorted(submission_list, key=lambda x:x[0])
        file = open("submissions/task_a.txt", "w")
        file.close()
        with open("submissions/task_a.txt", mode="w") as f:
            for i in range(len(submission_list)):
                for j in range(len(submission_list[i])):
                    if j < len(submission_list[i]) - 1:
                        f.write(f"{submission_list[i][j]}\t")
                    else:
                        f.write(f"{submission_list[i][j]}\n")
            print("succesfully wrote")
        read_file = pd.read_csv("submissions/task_a.txt")
        read_file.to_csv("submissions/task_a.csv", index=None)
        read_file.to_csv("submissions/task_a.tsv", index=None)
    else:
        model = XGBClassifier()
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
        model.fit(X_train, y_train, sample_weight=get_sample_weights(y_train))
        y_pred = model.predict(X_test)
        scores = []
        for pos_label in POSSIBLE_LABELS:
            score = f1_score(y_test, y_pred, average="macro", pos_label=pos_label)
            scores.append(score)
            print(pos_label, "-", score)
        print("task a:", mean(scores))


def task_b(texts, labels, model, label_columns, submit=False, submission_texts=None, file_paths=None):
    scores = []
    if submit:
        submission_list = [[file_path] for file_path in file_paths]
        predicted_labels_list = []

        for i in range(len(label_columns)):
            current_labels = labels[i]
            model = XGBClassifier()
            model.fit(texts, current_labels, sample_weight=get_sample_weights(current_labels))
            predicted_labels = model.predict(submission_texts)
            predicted_labels_list.append(predicted_labels)

        for i in range(len(predicted_labels_list)):
            for j in range(len(predicted_labels_list[i])):
                submission_list[j].append(predicted_labels_list[i][j])

        submission_list = sorted(submission_list, key=lambda x: x[0])

        file = open("submissions/task_b.txt", "w")
        file.close()
        with open("submissions/task_b.txt", mode="w") as f:
            for i in range(len(submission_list)):
                for j in range(len(submission_list[i])):
                    if j < len(submission_list[i]) - 1:
                        f.write(f"{submission_list[i][j]}\t")
                    else:
                        f.write(f"{submission_list[i][j]}\n")
            print("succesfully wrote")

        read_file = pd.read_csv("submissions/task_b.txt")
        read_file.to_csv("submissions/task_b.csv", index=None)
        read_file.to_csv("submissions/task_b.tsv", index=None)
    else:
        for i in range(len(label_columns)):
            X_train, X_test, y_train, y_test = train_test_split(texts, labels[i], test_size=0.2, random_state=13)
            model = XGBClassifier()
            model.fit(X_train, y_train, sample_weight=get_sample_weights(y_train))
            y_pred = model.predict(X_test)
            for pos_label in POSSIBLE_LABELS:
                score = f1_score(y_test, y_pred, average="weighted", pos_label=pos_label)
                scores.append(score)
                print(label_columns[i], "-", pos_label, "-", score)
        print("task b:", mean(scores))


def main():
    model = XGBClassifier()
    df = pd.read_csv("data/TRAINING/training_no_bad_lines.csv")
    print(len(df))
    print(df.describe())
    # exit()
    print(df.head())
    texts = df['Text Transcription'].to_list()

    cv = CountVectorizer()
    texts = cv.fit_transform(texts)
    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
    labels = [[] for _ in range(len(label_columns))]
    for i, label_column in enumerate(label_columns):
        for j in range(len(df)):
            labels[i].append(int(df.at[j, label_column]))

    test_df = pd.read_csv("data/test/Test.csv", delimiter='\t', error_bad_lines=False)
    test_df.to_csv("data/test/Test_no_bad_lines.csv")
    images_paths = test_df['file_name'].to_list()
    print(test_df.head())
    print(len(test_df))
    test_texts = test_df["Text Transcription"].to_list()
    test_texts = cv.transform(test_texts)
    submit = True

    task_a_labels = labels[0]
    task_a(texts, task_a_labels, model, submit=submit, submission_texts=test_texts, file_paths=images_paths)
    task_b(texts, labels, model, label_columns, submit=submit, submission_texts=test_texts, file_paths=images_paths)




if __name__=='__main__':
    main()