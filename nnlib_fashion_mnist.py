from functools import cache
import os
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from nnlib.activation import Activation_ReLU, Activation_Softmax
from nnlib.accuracy import Accuracy_Categorical
from nnlib.data import DataSet
from nnlib.layers import Layer_Dense
from nnlib.loss import Loss_CategoricalCrossentropy
from nnlib.model import Model
from nnlib.optimizers import Optimizer_Adam


class FashionMNistDataSet(DataSet):
    def __init__(self,
                 annotations_file: Path | str,
                 images_path: Path | str,
                 transform: Callable[[np.array], np.array] | None = None,
                 target_transform: Callable[[np.array], np.array] | None = None):
        """
        annotations_file: path to a csv file of:
            filename(png image), class_num (label)

        images_path: path to directory containing images

        transform: an optional function to modify the image data

        target_transform: an optional function to modify the labels
        """

        self.image_labels = pd.read_csv(annotations_file)
        self.images_dir = images_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    @cache
    def __getitem__(self, index: int) -> Tuple[np.array, np.int64]:
        """
        input: int index into image_labels

        returns: tuple(ndarray, long) representing the image data and label

        the image data and label may be transformed if transform and/or
        target_transform functions are defined

        results are cached, so first epoch will be slow, but others will
        be much faster, however I'm unsure if functools.cache keeps everything
        in memory or if the cache is backed by disk in some efficient manner
        """
        img_path = os.path.join(self.images_dir,
                                self.image_labels.iloc[index, 0])
        image = Image.open(img_path)
        image = np.array(image)
        label = self.image_labels.iloc[index, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def transform_image_data(image: np.array) -> np.array:
    """
    input: ndarray shape(1, 28, 28) dtype uint8 with values 0 -> 256
     - to -
    output: ndarray shape(768) dtype float32 with values -1 -> 1
    """
    # convert uint8 to float normalize values of 0 to 256 => -1 to 1
    image = (image.astype(np.float32) - 127.5) / 127.5
    # convert 28x28 matrix to len 768 vector
    image = image.flatten()
    return image


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
