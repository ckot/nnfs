import nnfs

from nnlib.data import DataLoader

from nnlib_fashion_mnist import (
    FashionMNistDataSet, transform_image_data, fashion_mnist_model as model
)

nnfs.init()

BATCH_SIZE = 128


train_dataset = FashionMNistDataSet("fashion_mnist/train/labels.csv",
                                    "fashion_mnist/train/images",
                                    transform=transform_image_data)

test_dataset = FashionMNistDataSet("fashion_mnist/test/labels.csv",
                                   "fashion_mnist/test/images",
                                   transform=transform_image_data)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=False)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, drop_last=False)


model.train(train_dataloader, validation_dataloader=test_dataloader,
            epochs=10, print_every=100)

model.save_parameters("nnlib_model_weights.pkl")
