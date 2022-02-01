import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from perceiver_pytorch import Perceiver
from process_data import load_train_data
from transformers import RobertaTokenizer, RobertaModel
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

from src.train_ml_model import load_computed_features


def instantiate_perceiver(input_channels: int = None, input_axis: int = None, num_classes: int = None):
    model = Perceiver(
        input_channels = input_channels or 1,          # number of channels for each token of the input
        input_axis = input_axis or 1,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 2,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 2,                   # depth of net. The shape of the final attention mechanism will be:
                                  #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = 2,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 4,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 2,            # number of heads for latent self attention, 8
        cross_dim_head = 4,         # number of dimensions per cross attention head
        latent_dim_head = 4,        # number of dimensions per latent self attention head
        num_classes = num_classes or 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
    )
    return model


from PIL import Image
import numpy as np
import pandas as pd

class MamiDataset(Dataset):
    def __init__(self):
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


from tqdm import tqdm

def train_model(train_dl, model):
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
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

from src.train_linear_transformer import instantiate_linear_attention_transformer
from src.train_performer import instantiate_performer
from src.train_sinkhorn_transformer import instantiate_sinkhorn_transformer
from src.train_reformer import instantiate_reformer

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 2
        transformer_option = "perceiver"
        if transformer_option == "perceiver":
            self.image_transformer = instantiate_perceiver(input_channels=1, input_axis=1)
            self.text_transformer = instantiate_perceiver(input_channels=1, input_axis=1)
            self.classif_transformer = instantiate_perceiver(input_channels=1, input_axis=1,  num_classes=num_classes)
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
        # import pdb
        # pdb.set_trace()
        concat_output = torch.reshape(input=concat_output, shape=(concat_output.shape[0], concat_output.shape[1] * concat_output.shape[2]))
        concat_output = torch.unsqueeze(concat_output, dim=-1)

        output = self.classif_transformer(concat_output)
        return output


def main(): # Killed
    mami_dataset = MamiDataset()
    train_size = int(0.8 * len(mami_dataset))
    test_size = len(mami_dataset) - train_size
    train, test = torch.utils.data.random_split(mami_dataset, [train_size, test_size])
    #
    # train, test = mami_dataset.get_splits()

    train_dl = DataLoader(train, batch_size=2, shuffle=True)
    test_dl = DataLoader(test, batch_size=2, shuffle=True)

    num_classes = 2

    model = TransformerModel()

    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # make a single prediction (expect class=1)

if __name__=="__main__":
    main()

