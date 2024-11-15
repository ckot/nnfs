import numpy as np
import nnfs

from fashion_mnist_dataset import create_data_mnist

from nnlib.accuracy import Accuracy_Categorical
from nnlib.activation import Activation_ReLU, Activation_Softmax
from nnlib.model import Model
from nnlib.layers import Layer_Dense
from nnlib.loss import Loss_CategoricalCrossentropy
from nnlib.optimizers import Optimizer_Adam

nnfs.init()


# Create dataset
X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# each image is a 28x28 matrix flateen each them to a 1d array of length 784
X = X.reshape(X.shape[0], -1)  # X.shape[0] is the number of samples
X_test = X_test.reshape(X_test.shape[0], -1)

# Scale features to floats between -1 and 1 . (currently uint8s between 0 and 255)
# subtract all values by mid value to get -mid_value and +mid_value and then
# divide by mid_value to normalize values to -1 and 1
#
# Another common technique os to scale the values to between 0 and 1
#
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


# Instantiate the model
model = Model()

# add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# set loss, optimizer and accuracy
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)


# Save the model
model.save('fashion_mnist.model')
