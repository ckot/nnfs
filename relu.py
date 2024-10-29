import numpy as np
import nnfs
nnfs.init()


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

