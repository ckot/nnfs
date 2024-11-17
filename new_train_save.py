import numpy as np
import nnfs

from nnlib.data import DataLoader

from fashion_mnist_dataset import (
    FashionMNistDataset, transform_image_data
)

from fashion_mnist_model import fashion_mnist_model as model

nnfs.init()

BATCH_SIZE = 128

train_dataset = FashionMNistDataset("fashion_mnist/train/labels.csv",
                                    "fashion_mnist/train/images",
                                    transform=transform_image_data)

test_dataset = FashionMNistDataset("fashion_mnist/test/labels.csv",
                                   "fashion_mnist/test/images",
                                   transform=transform_image_data)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, drop_last=False)



# Train the model
model.train(train_dataloader, validation_dataloader=test_dataloader,
            epochs=10, print_every=100)

model.save_parameters("fashion_mnist_weights.pkl")
