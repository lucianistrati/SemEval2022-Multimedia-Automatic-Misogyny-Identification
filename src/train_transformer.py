from torch import nn as nn
from src.train_ml_model import load_computed_features
# from src.train_transformer import TransformerModel
from src.train_linear_transformer import instantiate_linear_attention_transformer
from src.train_perceiver import instantiate_perceiver
from src.train_performer import instantiate_performer
from src.train_reformer import instantiate_reformer
from src.train_sinkhorn_transformer import instantiate_sinkhorn_transformer
from numpy import vstack
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch


class TransformerModel(nn.Module):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()
        num_classes = 2
        transformer_option = "perceiver"
        if transformer_option == "perceiver":
            self.image_transformer = instantiate_perceiver(input_channels=1, input_axis=1)
            self.text_transformer = instantiate_perceiver(input_channels=1, input_axis=1)
            self.classif_transformer = instantiate_perceiver(input_channels=1, input_axis=1, num_classes=num_classes)
        elif transformer_option == "reformer":
            self.image_transformer = instantiate_reformer(input_channels=1)
            self.text_transformer = instantiate_reformer(input_channels=1)
            self.classif_transformer = instantiate_reformer(num_classes=num_classes)
        elif transformer_option == "performer":
            self.image_transformer = instantiate_performer(input_channels=1)
            self.text_transformer = instantiate_performer(input_channels=1)
            self.classif_transformer = instantiate_performer(num_classes=num_classes)
        elif transformer_option == "sinkhorn_transformer":
            self.image_transformer = instantiate_sinkhorn_transformer(input_channels=1)
            self.text_transformer = instantiate_sinkhorn_transformer(input_channels=1)
            self.classif_transformer = instantiate_sinkhorn_transformer(num_classes=num_classes)
        elif transformer_option == "linear_attention_transformer":
            self.image_transformer = instantiate_linear_attention_transformer(input_channels=1)
            self.text_transformer = instantiate_linear_attention_transformer(input_channels=1)
            self.classif_transformer = instantiate_linear_attention_transformer(num_classes=num_classes)
        else:
            raise Exception("wrong transformer_option given!")

    def forward(self, batch):
        image_batch, text_batch = batch[:, :768], batch[:, 768:]
        image_batch = torch.unsqueeze(image_batch, dim=-1)
        text_batch = torch.unsqueeze(text_batch, dim=-1)
        image_output = self.image_transformer(image_batch)
        text_output = self.text_transformer(text_batch)
        concat_output = torch.stack([image_output, text_output])
        concat_output = torch.reshape(input=concat_output,
                                      shape=(concat_output.shape[0], concat_output.shape[1] * concat_output.shape[2]))
        concat_output = torch.unsqueeze(concat_output, dim=-1)

        output = self.classif_transformer(concat_output)
        return output


class MamiDataset(Dataset):
    """

    """
    def __init__(self):
        """

        """
        df = pd.read_csv("data/TRAINING_csvs/training_no_bad_lines.csv")
        test_df = pd.read_csv("data/test_csvs/Test_no_bad_lines.csv")

        train_filenames = df["file_name"].to_list()
        test_filenames = test_df['file_name'].to_list()

        X_train_text, X_test_text = load_computed_features(train_filenames,
                                                           test_filenames,
                                                           data_type="text")

        X_train_vision, X_test_vision = load_computed_features(train_filenames,
                                                               test_filenames, data_type="vision")

        X_train = np.hstack((X_train_text, X_train_vision))
        X_test = np.hstack((X_test_text, X_test_vision))
        print(len(X_test))

        label_columns = ["misogynous", "shaming", "stereotype", "objectification", "violence"]

        labels = [[] for _ in range(len(label_columns))]
        for i, label_column in enumerate(label_columns):
            for j in range(len(df)):
                labels[i].append(int(df.at[j, label_column]))

        idx = 0
        y_train = labels[idx]
        self.X, self.y = X_train, y_train

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)


def train_model(train_dl, model):
    """

    :param train_dl:
    :param model:
    :return:
    """
    # define the optimization
    criterion = CrossEntropyLoss()
    num_epochs = 10
    optimizer = Adam(model.parameters())
    # enumerate epochs
    for epoch in tqdm(range(num_epochs)):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        # evaluate the model


def evaluate_model(test_dl, model):
    """

    :param test_dl:
    :param model:
    :return:
    """
    predictions, actuals = list(), list()
    for i, (inputs, targets) in tqdm(enumerate(test_dl)):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, np.argmax(predictions, axis=-1))
    return acc


# make a class prediction for one row of data
def predict(row, model):
    """

    :param row:
    :param model:
    :return:
    """
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


def main():  # Killed
    mami_dataset = MamiDataset()
    train_size = int(0.8 * len(mami_dataset))
    test_size = len(mami_dataset) - train_size
    train, test = torch.utils.data.random_split(mami_dataset, [train_size, test_size])
    #
    # train, test = mami_dataset.get_splits()

    train_dl = DataLoader(train, batch_size=2, shuffle=True)
    test_dl = DataLoader(test, batch_size=2, shuffle=True)

    model = TransformerModel()

    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # make a single prediction (expect class=1)


if __name__ == "__main__":
    main()
