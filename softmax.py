import numpy as np
import nnfs
nnfs.init()

class Activation_Softmax:

    def forward(self, inputs):
        # calculate unnormalized probs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize the probs for each sample
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs

