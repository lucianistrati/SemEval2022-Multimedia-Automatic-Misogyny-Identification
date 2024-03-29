from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from heat_map import plot_heatmap
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import check_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pdb


EPSILON = 1e-10


def empty_classif_loggings():
    """

    :return:
    """
    return [['F1', 0.0], ['Accuracy', 0.0], ['Normalized_confusion_matrix',
                                             0.0]]


def empty_regress_loggings():
    """

    :return:
    """
    return [['R2', 0.0], ['MAPE', 0.0], ['MAE', 0.0], ['MSE', 0.0], ['MDA',
                                                                     0.0],
            ['MAD', 0.0]]


def mean_absolute_percentage_error(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def get_regress_perf_metrics(y_test, y_pred, model_name="",
                             target_feature="",
                             logging_metrics_list=empty_regress_loggings(),
                             visualize_metrics=False):
    """

    :param y_test:
    :param y_pred:
    :param model_name:
    :param target_feature:
    :param logging_metrics_list:
    :param visualize_metrics:
    :return:
    """
    if visualize_metrics:
        print("For " + model_name + " regression algorithm the following "
                                    "performance metrics were determined:")

    if target_feature == 'all':
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == "MSE":
            logging_metrics_list[i][1] = str(mean_squared_error(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAE":
            logging_metrics_list[i][1] = str(mean_absolute_error(y_test,
                                                                 y_pred))
        elif logging_metrics_list[i][0] == "R2":
            logging_metrics_list[i][1] = str(r2_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAPE":
            logging_metrics_list[i][1] = str(mape(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MDA":
            logging_metrics_list[i][1] = str(mda(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAD":
            logging_metrics_list[i][1] = 0.0

    if visualize_metrics:
        print("MSE: ", mean_squared_error(y_test, y_pred))
        print("MAE: ", mean_absolute_error(y_test, y_pred))
        print("R squared score: ", r2_score(y_test, y_pred))
        print("Mean absolute percentage error:", mape(y_test, y_pred))
        try:
            print("Mean directional accuracy:", mda(y_test, y_pred))
        except TypeError:
            print("Type error", model_name)

    return logging_metrics_list


def get_classif_perf_metrics(y_test, y_pred, model_name="",
                             logging_metrics_list=empty_classif_loggings(), num_classes=1):
    """

    :param y_test:
    :param y_pred:
    :param model_name:
    :param logging_metrics_list:
    :param num_classes:
    :return:
    """
    for model_categoy in ["FeedForward", "Convolutional", "LSTM"]:
        if model_categoy in model_name:
            y_pred = np.array([np.argmax(pred) for pred in y_pred])
            y_test = np.array([np.argmax(pred) for pred in y_test])
    print("For " + model_name + " classification algorithm the following "
                                "performance metrics were determined on the "
                                "test set:")
    number_of_classes = num_classes
    print("NUM CLASSES", number_of_classes)

    if number_of_classes == 2:
        for i in range(len(logging_metrics_list)):
            if logging_metrics_list[i][0] == 'Accuracy':
                logging_metrics_list[i][1] = str(accuracy_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'Precision':
                logging_metrics_list[i][1] = str(precision_score(y_test,
                                                                 y_pred))
            elif logging_metrics_list[i][0] == 'Recall':
                logging_metrics_list[i][1] = str(recall_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'F1':
                logging_metrics_list[i][1] = str(f1_score(y_test, y_pred,
                                                          average='weighted'))
        print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 2)))
        print("Precision: " + str(precision_score(y_test, y_pred,
                                                  average='weighted')))
        print("Recall: " + str(recall_score(y_test, y_pred,
                                            average='weighted')))
    else:
        for i in range(len(logging_metrics_list)):
            if logging_metrics_list[i][0] == 'Accuracy':
                logging_metrics_list[i][1] = str(accuracy_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'Precision':
                logging_metrics_list[i][1] = str(precision_score(y_test,
                                                                 y_pred))
            elif logging_metrics_list[i][0] == 'Recall':
                logging_metrics_list[i][1] = str(recall_score(y_test, y_pred))
            elif logging_metrics_list[i][0] == 'F1':
                # import pdb
                # pdb.set_trace()
                logging_metrics_list[i][1] = str(f1_score(y_test, y_pred,
                                                          average='weighted'))
            elif logging_metrics_list[i][0] == 'Classification_report':
                logging_metrics_list[i][1] = str(
                    classification_report(y_test, y_pred, digits=2))

        print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 2)))
        print("Precision: " + str(precision_score(y_test, y_pred,
                                                  average='weighted')))
        print("Recall: " + str(recall_score(y_test, y_pred,
                                            average='weighted')))

        print("Classification report: \n" + str(
            classification_report(y_test, y_pred, digits=2)))

    C = confusion_matrix(y_test, y_pred)
    if number_of_classes <= 3:
        print("Confusion matrix:\n", C)
        print("Normalized confusion matrix:\n",
              np.around(C / C.astype(np.float).sum(axis=1), decimals=2))

        for i in range(len(logging_metrics_list)):
            if logging_metrics_list[i][0] == 'Confusion_matrix':
                logging_metrics_list[i][1] = np.array2string(np.around(C,
                                                                       decimals=2),
                                                             precision=2,
                                                             separator=',',
                                                             suppress_small=True)
            elif logging_metrics_list[i][0] == 'Normalized_confusion_matrix':
                logging_metrics_list[i][1] = np.array2string(np.around(C / C.astype(np.float).sum(axis=1),
                                                                       decimals=2), precision=2,
                                                             separator=',', suppress_small=True)

    return logging_metrics_list


def plot_loss_and_acc(history):
    """

    :param history:
    :return:
    """
    plt.plot(history.history["acc"], label="train")
    plt.plot(history.history["val_acc"], label="test")

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "test"])
    plt.show()


def plot_multiple_metrics(history, model_name=""):
    """

    :param history:
    :param model_name:
    :return:
    """
    # import pdb
    # pdb.set_trace()
    keys = list(history.history.keys())
    colors = ['g', 'b', 'r', 'y', 'p']
    for i in range(len(keys)):
        hist_key = keys[i]
        metric = history.history[hist_key]
        actual_num_epochs = range(1, len(metric) + 1)
        plt.plot(actual_num_epochs, metric, colors[i], label=hist_key)
    if model_name:
        plt.title("Metrics obtained for " + model_name)
    plt.legend()
    plt.show()


def plot_metric(history, metric_name, model_name):
    """

    :param history:
    :param metric_name:
    :param model_name:
    :return:
    """
    if "acc" in history.history.keys() or "accuracy" in history.history.keys():
        if "accuracy" in history.history.keys():
            metric = history.history["accuracy"]
            val_metric = history.history["val_accuracy"]
        else:
            metric = history.history[metric_name]
            val_metric = history.history["val_" + metric_name]
    else:
        metric = history.history[metric_name]
        val_metric = history.history["val_" + metric_name]

    actual_num_epochs = range(1, len(metric) + 1)

    plt.plot(actual_num_epochs, metric, "g",
             label="Train " + metric_name + " for " + model_name)
    plt.plot(actual_num_epochs, val_metric, "b",
             label="Val " + metric_name + " for " + model_name)
    plt.legend()
    plt.title(metric_name.capitalize() + " for " + model_name)
    plt.show()


def train_nn(X_train, y_train, X_test, y_test, model_name, num_classes, class_weight=None, backbone_option="linear"):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param model_name:
    :param num_classes:
    :param class_weight:
    :param backbone_option:
    :return:
    """
    early_stopping = EarlyStopping(
        patience=5,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )

    lr_schedule = ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )

    n_epochs = 100
    learning_rate = 1e-3

    if num_classes > 1:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    if num_classes > 1:
        activation_fn = "softmax"
    else:
        activation_fn = "linear"

    if backbone_option == "linear":
        classifier = tf.keras.models.Sequential()
        classifier.add(tf.keras.layers.Dense(256, input_shape=X_test[0].shape, activation="relu"))
        classifier.add(tf.keras.layers.Dense(256, activation="relu"))
        classifier.add(tf.keras.layers.Dense(num_classes, activation=activation_fn))
    elif backbone_option == "convolutional":
        classifier = tf.keras.models.Sequential()
        classifier.add(tf.keras.layers.Dense(256, input_shape=X_test[0].shape, activation="relu"))
        classifier.add(tf.keras.layers.Dense(256, activation="relu"))
        classifier.add(tf.keras.layers.Dense(num_classes, activation=activation_fn))
    else:
        raise Exception("wrong backbone given")

    adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
    if num_classes > 1:
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        metrics_function = 'accuracy'
    else:
        loss_function = tf.keras.losses.MeanSquaredError()
        metrics_function = 'mae'

    classifier.compile(optimizer=adam_opt, loss=loss_function,
                       metrics=[metrics_function])

    if class_weight is not None:
        history = classifier.fit(X_train,
                                 y_train,
                                 epochs=n_epochs,
                                 verbose=2,
                                 class_weight=class_weight,
                                 callbacks=[early_stopping, lr_schedule])
    else:
        history = classifier.fit(X_train,
                                 y_train,
                                 epochs=n_epochs,
                                 verbose=2,
                                 callbacks=[early_stopping, lr_schedule])

    y_pred = classifier.predict(X_test)

    print(classifier.summary())

    if num_classes > 1:
        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

    plot_heatmap(y_pred, y_test)
    logging_metrics_list = get_classif_perf_metrics(y_test,
                                                    y_pred,
                                                    model_name=model_name, num_classes=num_classes)

    print(logging_metrics_list)

    plot_multiple_metrics(history)
