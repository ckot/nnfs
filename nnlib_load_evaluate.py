
from nnlib.data import DataLoader

from nnlib_fashion_mnist import (
    FashionMNistDataSet, transform_image_data, fashion_mnist_model as model
)

test_dataset = FashionMNistDataSet("fashion_mnist/test/labels.csv",
                                   "fashion_mnist/test/images",
                                   transform=transform_image_data)

test_dataloader = DataLoader(test_dataset)


model.load_parameters("nnlib_model_weights.pkl")

model.evaluate(test_dataloader)