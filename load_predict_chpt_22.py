import numpy as np
import nnfs

from fashion_mnist_dataset import create_data_mnist

from nnlib.model import Model

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



# Load the model
model = Model.load('fashion_mnist.model')

# Evaluate the model
model.evaluate(X_test, y_test)