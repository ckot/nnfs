import os
from functools import cache
from pathlib import Path
from typing import Callable
import sys

import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image


class CustomFashionMNistDataset(Dataset):
    def __init__(self,
                 annotations_file: Path | str,
                 images_path: Path | str,
                 transform: Callable[[Tensor], Tensor] | None = None,
                 target_transform: Callable[[Tensor], Tensor] | None = None):
        self.image_labels = pd.read_csv(annotations_file)
        self.images_dir = images_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    # @cache
    def __getitem__(self, index: int):
        # print index to verify shuffle is working
        # print(f"fetching index: {index} of {len(self) + 1}")
        img_path = os.path.join(self.images_dir,
                                self.image_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

flatten = nn.Flatten()

def transform_image_data(image: Tensor) -> Tensor:
    """
    input: Tensor shape(batchsize, 1, 28, 28) dtype uint8 with values 0 -> 255
     - to -
    output: Tensor shape(batchsize, 768) dtype float32 with values -1 -> 1
    """
    # convert uint8 to float
    image.to(dtype=torch.float32)
    # normalize values of 0 to 256 => -1 to 1
    # image = (image - 127.5) / 127.5
    image = image / 255
    # convert 28x28 matrix to len 768 vector
    # image = flatten(image)
    return image


def one_hot_encode(label: Tensor) -> Tensor:
    # couldn't figure out how to use functional.one_hot()
    # one_hot_encoded = torch.nn.functional.one_hot(label, num_classes=10)
    idx = int(label)
    encoded = torch.zeros(10, dtype=torch.long)
    encoded[idx] = 1
    return encoded


# lbl = torch.Tensor([2])
# one_hot = one_hot_encode(lbl)
# print(one_hot)
# sys.exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# #Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

training_data = CustomFashionMNistDataset("fashion_mnist/train/labels.csv",
                                          "fashion_mnist/train/images",
                                          transform=transform_image_data,
                                          target_transform=None)
                                        #   target_transform=one_hot_encode)
test_data = CustomFashionMNistDataset("fashion_mnist/test/labels.csv",
                                      "fashion_mnist/test/images",
                                      transform=transform_image_data,
                                      target_transform=None)
                                    #   target_transform=one_hot_encode)



BATCH_SIZE = 128
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=False)


# for X, y in train_dataloader:
#     print(f"Shape of X [N, C, W, H]: {X.shape}")
#     print(f"Shape of y: [N] {y.shape} {y.dtype}")
#     break
# sys.exit(0)

mdl = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(X.shape, X.dtype, y.shape, y.dtype)
        # print(X[0])
        # sys.exit(0)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, mdl, loss_fn, optimizer)
    test(test_dataloader, mdl, loss_fn)
print("Done!")

torch.save(mdl.state_dict(), "model.pth")