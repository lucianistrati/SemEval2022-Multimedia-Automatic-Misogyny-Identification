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
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


def instantiate_perceiver(input_channels: int = None, num_classes: int = None):
    model = Perceiver(
        input_channels = input_channels or 3,          # number of channels for each token of the input
        input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 2,                   # depth of net. The shape of the final attention mechanism will be:
                                  #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = 8,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 16,            # latent dimension
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
    # img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized
    # model(img) # (1, 1000)
    return model

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def embed_text(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # import pdb
    # pdb.set_trace()
    return output.pooler_output

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocessing = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

from PIL import Image

class MamiDataset(Dataset):
    def __init__(self):
        train_images, train_texts, train_labels = load_train_data("data/train_numpy_arrays")
        self.X, self.y = [], []

        for (image, text) in zip(train_images, train_texts):
            encoded_text = embed_text(text)
            image = preprocessing(Image.fromarray(image))
            datapoint = (image, encoded_text)
            # datapoint = torch.stack((encoded_text, image))
            # import pdb
            # pdb.set_trace()
            self.X.append(datapoint)

        for label in train_labels:
            self.y.append(label)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)



def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
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
    for i, (inputs, targets) in enumerate(test_dl):
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
    acc = accuracy_score(actuals, predictions)
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


class PerceiverModel(nn.Module):
    def __init__(self, num_classes):
        self.image_perceiver = instantiate_perceiver(input_channels=3)
        self.text_perceiver = instantiate_perceiver(input_channels=1)
        self.classif_perceiver = instantiate_perceiver(num_classes=num_classes)

    def __forward__(self, batch):
        image_batch, text_batch = batch
        image_output = self.image_perceiver(image_batch)
        text_output = self.text_perceiver(text_batch)
        concat_output = torch.stack([image_output, text_output])
        output = self.classif_perceiver(concat_output)
        return output


def main():
    mami_dataset = MamiDataset()

    train, test = mami_dataset.get_splits()

    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)

    num_classes = 2

    model = PerceiverModel(num_classes=num_classes)

    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # make a single prediction (expect class=1)

if __name__=="__main__":
    main()

