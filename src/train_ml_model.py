from statistics import mean

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# SVC fintuning links:
# https://link.springer.com/chapter/10.1007/978-3-540-39857-8_33
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
# https://www.researchgate.net/publication/344386194_HOW_TO_FINE-TUNE_SUPPORT_VECTOR_MACHINES_FOR_CLASSIFICATION

POSSIBLE_LABELS = [0, 1]


def get_sample_weights(y_train):
    from sklearn.utils import class_weight
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    return classes_weights


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def load_computed_features(train_filenames, test_filenames, data_type="text"):
    path = "data/online_computed_numpy_arrays"
    image_type = "po"

    if data_type.startswith("text"):
        # X_train = np.load("data/online_computed_numpy_arrays/train_sentence_transformer_multi_qa_mpnet.npy", allow_pickle=True)
        # X_test = np.load("data/online_computed_numpy_arrays/test_sentence_transformer_multi_qa_mpnet.npy", allow_pickle=True)
        # X_train_1 = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[-1]))
        # X_test_1 = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[-1]))

        # X_train = np.load("data/online_computed_numpy_arrays/train_sentence_transformer.npy", allow_pickle=True)
        # X_test = np.load("data/online_computed_numpy_arrays/test_sentence_transformer.npy", allow_pickle=True)
        # X_train_2 = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[-1]))
        # X_test_2 = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[-1]))

        X_train = np.load("data/online_computed_numpy_arrays/train_roberta_distil.npy", allow_pickle=True)
        X_test = np.load("data/online_computed_numpy_arrays/test_roberta_distil.npy", allow_pickle=True)
        X_train_3 = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[-1]))
        X_test_3 = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[-1]))

        X_train_text = X_train_3  # , X_train_4))
        X_test_text = X_test_3  # , X_test_4))

        # X_train = np.load("data/online_computed_numpy_arrays/train_roberta.npy", allow_pickle=True)
        # X_test = np.load("data/online_computed_numpy_arrays/test_roberta.npy", allow_pickle=True)
        # X_train_4 = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[-1]))
        # X_test_4 = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[-1]))

        # X_train_text = np.hstack((X_train_2, X_train_3))#, X_train_4))
        # X_test_text = np.hstack((X_test_2, X_test_3))#, X_test_4))

    if data_type.endswith("vision"):
        sorted_filenames_train = np.load("data/online_computed_numpy_arrays/train_image_filenames.npy", allow_pickle=True)
        sorted_filenames_test = np.load("data/online_computed_numpy_arrays/test_image_filenames.npy", allow_pickle=True)

        assert len(sorted_filenames_test) == len(test_filenames)
        assert len(sorted_filenames_train) == len(train_filenames)

        if image_type == "hs":
            X_train = np.load("data/online_computed_numpy_arrays/train_tiny_vit_features_hs_10K.npy", allow_pickle=True)
            X_test = np.load("data/online_computed_numpy_arrays/test_tiny_vit_features_hs.npy", allow_pickle=True)

            X_train = np.array([X.detach().cpu().numpy() for X in X_train])
            X_test = np.array([X.detach().cpu().numpy() for X in X_test])

            X_train = np.reshape(X_train, newshape=(X_train.shape[0], 197 * 32))
            X_test = np.reshape(X_test, newshape=(X_test.shape[0], 197 * 32))
        elif image_type == "po":
            X_train = np.load("data/online_computed_numpy_arrays/train_tiny_vit_features_po_10K.npy", allow_pickle=True)
            X_test = np.load("data/online_computed_numpy_arrays/test_tiny_vit_features_po.npy", allow_pickle=True)

            X_train = np.array([X.detach().cpu().numpy() for X in X_train])
            X_test = np.array([X.detach().cpu().numpy() for X in X_test])

            X_train = np.reshape(X_train, newshape=(X_train.shape[0], 32))
            X_test = np.reshape(X_test, newshape=(X_test.shape[0], 32))
        else:
            raise Exception("wrong")

        new_X_train = []
        new_X_test = []

        sorted_filenames_train = list(sorted_filenames_train)
        sorted_filenames_test = list(sorted_filenames_test)

        for i, file_name in enumerate(train_filenames):
            idx = sorted_filenames_train.index(file_name)
            new_X_train.append(X_train[idx])

        for i, file_name in enumerate(test_filenames):
            idx = sorted_filenames_test.index(file_name)
            new_X_test.append(X_test[idx])

        X_train_vision = np.array(new_X_train)
        X_test_vision = np.array(new_X_test)

    if data_type == "text":
        return X_train_text, X_test_text
    elif data_type == "vision":
        return X_train_vision, X_test_vision
    elif data_type == "text_vision":
        return np.hstack((X_train_text, X_train_vision)), \
               np.hstack((X_test_text, X_test_vision))


def task_b(texts, labels, model, label_columns, submit=False, submission_texts=None, file_paths=None, embedding_model="", X_train=None, X_test=None):
    if X_train is not None:
        if embedding_model == "tfidfvectorizer":
            texts = texts.toarray()  # take a lot of time
            texts = np.hstack((texts, X_train))
            texts = texts.astype(np.float16)
        else:
            texts = X_train

    if X_test is not None:
        if embedding_model == "tfidfvectorizer":
            submission_texts = submission_texts.toarray()
            submission_texts = np.hstack((submission_texts, X_test))
            submission_texts = submission_texts.astype(np.float16)
        else:
            submission_texts = X_test
    print(embedding_model)
    scores = []
    best_task_a_score = 1e9
    best_task_a_model = ""

    best_task_b_score = 1e9
    best_task_b_model = ""
    if submit:
        submission_list = [[file_path] for file_path in file_paths]
        predicted_labels_list = []

        for i in range(len(label_columns)):
            current_labels = labels[i]
            if embedding_model == "count_vectorizer":
                model = RandomForestClassifier(class_weight="balanced")
                print("random forest")
            elif embedding_model == "tfidfvectorizer":
                print("svc")
                model = SVC(class_weight="balanced")  # , degree=9)
            elif embedding_model == "computed_features":
                model = SVC(class_weight="balanced")
                print("xgb")
            else:
                raise Exception("WRONG EMBEDDING Model")
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
        if embedding_model == "count_vectorizer":
            model = RandomForestClassifier(class_weight="balanced")
            print("random forest")
        elif embedding_model == "tfidfvectorizer":  # knn classifier
            model = SVC(class_weight="balanced")
            print("svc")
        elif embedding_model == "computed_features":
            model = SVC(class_weight="balanced")
            print("xgb")
        else:
            raise Exception("WRONG EMBEDDING Model")
        try:
            model.class_weight = "balanced"
        except:
            pass
        scores = []
        for i in range(len(label_columns)):
            X_train, X_test, y_train, y_test = train_test_split(texts, labels[i], test_size=0.2, random_state=13)
            try:
                model.fit(X_train, y_train, sample_weight=get_sample_weights(y_train))
            except:
                model.fit(X_train, y_train)
                pass
            y_pred = model.predict(X_test)
            for pos_label in POSSIBLE_LABELS:
                score = f1_score(y_test, y_pred, average="weighted", pos_label=pos_label)
                scores.append(score)
                print(label_columns[i], "-", pos_label, "-", score)
        print("*" * 30)
        # print("model_name: ", model_name)
        print("task b:", mean(scores))
        print("task a:", mean(scores[:2]))
        print("*" * 30)
        # if mean(scores) > best_task_b_score:
        #     best_task_b_score = mean(scores)
        #     best_task_b_model = model_name
        #
        # if mean(scores[:2]) > best_task_a_score:
        #     best_task_a_score = mean(scores[:2])
        #     best_task_a_model = model_name

    # print("B:", best_task_b_model, best_task_b_score)
    # print("A:", best_task_a_model, best_task_a_score)


def extract_features(text):
    return None


from tqdm import tqdm

from src.text_preprocess import embed_text


def main():
    embedding_idx = -1
    embedding_model = ["count_vectorizer", "tfidfvectorizer", "handmade_features", "text_embedding", "computed_features"][embedding_idx]
    embedding_method = "bert_large"

    print(embedding_model)

    vectorizer = None
    if embedding_model == "count_vectorizer":
        vectorizer = CountVectorizer()
    elif embedding_model == "tfidfvectorizer":
        vectorizer = TfidfVectorizer(max_features=15000)
    # https://stackoverflow.com/questions/44066264/how-to-choose-parameters-in-tfidfvectorizer-in-sklearn-during-unsupervised-clust

    # model = XGBClassifier()
    df = pd.read_csv("data/TRAINING_csvs/training_no_bad_lines.csv")
    print(len(df))
    print(df.describe())

    # exit()

    def flatten(t):
        return [item for sublist in t for item in sublist]

    print(df.head())
    texts = df['Text Transcription'].to_list()
    train_file_names = df['file_name'].to_list()
    texts = np.load("data/online_computed_numpy_arrays/train_image_features_gpt2.npy").tolist()
    texts = flatten(texts)
    sorted_filenames_train = np.load("data/online_computed_numpy_arrays/train_image_filenames.npy", allow_pickle=True).tolist()
    new_texts = []
    for i, file_name in enumerate(train_file_names):
        idx = sorted_filenames_train.index(file_name)
        new_texts.append(texts[idx])
    texts_captions = new_texts
    texts = df['Text Transcription'].to_list()
    texts = [caption + " " + test_text for (caption, test_text) in zip(texts_captions, texts)]

    if vectorizer is not None:
        texts = vectorizer.fit_transform(texts)

    label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
    labels = [[] for _ in range(len(label_columns))]
    for i, label_column in enumerate(label_columns):
        for j in range(len(df)):
            labels[i].append(int(df.at[j, label_column]))

    test_df = pd.read_csv("data/test_csvs/Test.csv", delimiter='\t', error_bad_lines=False)
    test_df.to_csv("data/test_csvs/Test_no_bad_lines.csv")

    images_paths = test_df['file_name'].to_list()

    test_texts = test_df["Text Transcription"].to_list()
    test_file_names = test_df["file_name"].to_list()
    test_texts = np.load("data/online_computed_numpy_arrays/test_images_features_gpt2.npy").tolist()
    test_texts = flatten(test_texts)
    sorted_filenames_test = np.load("data/online_computed_numpy_arrays/test_image_filenames.npy", allow_pickle=True).tolist()
    new_test_texts = []
    for i, file_name in enumerate(test_file_names):
        idx = sorted_filenames_test.index(file_name)
        new_test_texts.append(test_texts[idx])
    test_texts_captions = new_test_texts

    test_texts = test_df["Text Transcription"].to_list()

    test_texts = [caption + " " + test_text for (caption, test_text) in zip(test_texts_captions, test_texts)]

    if vectorizer is not None:
        test_texts = vectorizer.transform(test_texts)
    else:
        if embedding_model == "text_embedding":
            embedded_texts = []
            for text in tqdm(test_texts):
                embedded_texts.append(embed_text(text, embedding_model=embedding_method))
            test_texts = embedded_texts
            # if os.path.exists("data/texts_" + embedding_method + ".npy"):
            np.save(file="data/texts_" + embedding_method + ".npy",
                    arr=np.array(test_texts),
                    allow_pickle=True)
            a = np.load(file="data/texts_" + embedding_method + ".npy", allow_pickle=True)
            print(a.shape)
            print(a[0].shape)
            # exit()
        elif embedding_model == "paper_features":
            test_texts = [embed_text(text, embedding_model="paper_features") for text in test_texts]
        elif embedding_model == "computed_features":
            pass
        else:
            raise Exception("wrong embedding model!")

    submit = False

    # task_a_labels = labels[0]
    # task_a(texts, task_a_labels, model, submit=submit, submission_texts=test_texts, file_paths=images_paths)

    train_filenames = df["file_name"].to_list()
    test_filenames = test_df['file_name'].to_list()

    X_train = None
    X_test = None
    X_train, X_test = load_computed_features(train_filenames, test_filenames, data_type="text")

    task_b(texts, labels, None, label_columns, submit=submit, submission_texts=test_texts,
           file_paths=images_paths, embedding_model=embedding_model, X_train=X_train, X_test=X_test)
    print(embedding_model)
    print("gpt2 image captioning")


if __name__ == '__main__':
    main()
"""
de gandit intrebari pentru VQA:
is this offensive? is this objectification? is this misogynous?
"""
