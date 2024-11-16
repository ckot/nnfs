from nnlib.model import Model
from nnlib.layers import Layer_Dense
from nnlib.activation import Activation_ReLU, Activation_Softmax
from nnlib.accuracy import Accuracy_Categorical
from nnlib.loss import Loss_CategoricalCrossentropy
from nnlib.optimizers import Optimizer_Adam


# Instantiate the model
fashion_mnist_model = Model()

# add layers
fashion_mnist_model.add(Layer_Dense(28*28, 128))
fashion_mnist_model.add(Activation_ReLU())
fashion_mnist_model.add(Layer_Dense(128, 128))
fashion_mnist_model.add(Activation_ReLU())
fashion_mnist_model.add(Layer_Dense(128, 10))
fashion_mnist_model.add(Activation_Softmax())

# set loss, optimizer and accuracy
fashion_mnist_model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
fashion_mnist_model.finalize()
