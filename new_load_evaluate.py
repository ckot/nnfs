import numpy as np

from fashion_mnist_dataset import (
    DataLoader, FashionMNistDataset, transform_image_data
)
from fashion_mnist_model import fashion_mnist_model as model
# from nnlib.model import Model
# from nnlib.layers import Layer_Dense, Layer_Input
# from nnlib.activation import Activation_ReLU, Activation_Softmax
# from nnlib.loss import Loss, Loss_CategoricalCrossentropy
# from nnlib.enhancements import Activation_Softmax_Loss_CategoricalCrossentropy
# from nnlib.optimizers import Optimizer_Adam
# from nnlib.accuracy import Accuracy_Categorical

test_dataset = FashionMNistDataset("fashion_mnist/test/labels.csv",
                                   "fashion_mnist/test/images",
                                   transform=transform_image_data)

test_dataloader = DataLoader(test_dataset)


model.load_parameters("fashion_mnist_weights.pkl")

model.evaluate(test_dataloader)