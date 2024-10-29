import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

from dense_layer import Layer_Dense
from relu import Activation_ReLU
from softmax import Activation_Softmax
from loss import Loss_CategoricalCrossentropy

# create dataset
X, y = spiral_data(samples=100, classes=3)

# dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# use relue activation for layer 1
activation1 = Activation_ReLU()

# 2nd Dense layer with 3 inputs (output of layer 1) and
# 3 output values
dense2 = Layer_Dense(3,3)
# use softmax activation for layer 2
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# do forward pass of training data through layer 1
dense1.forward(X)
# perform the activation of layer1
activation1.forward(dense1.output)

# do forward pass of 2nd layer taking the activations of layer1 as input
dense2.forward(activation1.output)
# perform the activation of layer2
activation2.forward(dense2.output)
print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)
print(predictions[:5])
print(y[:5])

accuruacy = np.mean(predictions == y)
print('acc:', accuruacy)


# print(y[:5])

