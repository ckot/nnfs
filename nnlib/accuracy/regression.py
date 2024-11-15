import numpy as np

from .base import Accuracy

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    # Calculates precision value
    # based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
