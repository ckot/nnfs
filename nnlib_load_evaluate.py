import numpy as np

from nnlib.data import DataLoader

from fashion_mnist_dataset import (
    FashionMNistDataset, transform_image_data
)
from fashion_mnist_model import fashion_mnist_model as model

test_dataset = FashionMNistDataset("fashion_mnist/test/labels.csv",
                                   "fashion_mnist/test/images",
                                   transform=transform_image_data)

test_dataloader = DataLoader(test_dataset)


model.load_parameters("nnlib_model_weights.pkl")

model.evaluate(test_dataloader)