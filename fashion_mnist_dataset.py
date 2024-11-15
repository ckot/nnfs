import os
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # for each label folder:
    for label in tqdm(labels):
        for file in tqdm(os.listdir(os.path.join(path, dataset, label))):
            # Read the image
            pic = Image.open(os.path.join(path, dataset, label, file))
            pixels = np.array(pic)
            # And append it and a label to the lists
            X.append(pixels)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):

    # Load both sets separately
    print("loading training data")
    X, y = load_mnist_dataset("train", path)
    print("loading test data")
    X_test, y_test = load_mnist_dataset("test", path)

    # And return all the data
    return X, y, X_test, y_test


class FashionMNistDataset:
    """
    annotations_file: path to a csv file of filename(png image), class_num (label)

    images_path: path to directory containing images

    transform: an optional function to modify the image data

    target_transform: an optional function to modify the labels
    """
    def __init__(self,
                 annotations_file: Path | str,
                 images_path: Path | str,
                 transform: Callable[[np.array], np.array] | None = None,
                 target_transform: Callable[[np.array], np.array] | None = None):
        self.image_labels = pd.read_csv(annotations_file)
        self.images_dir = images_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index: int):
        """
        input: int index into image_labels

        returns: tuple(ndarray, long) representing the image data and label

        the image data and label may be transformed if transform and/or
        target_transform functions are defined
        """
        img_path = os.path.join(self.images_dir,
                                self.image_labels.iloc[index, 0])
        image = Image.open(img_path)
        image = np.array(image)
        label = self.image_labels.iloc[index, 1]
        #print(f"fetching index {index}: {img_path} label: {label} {type(label)} {label.dtype}")

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


class DataLoader:
    def __init__(self, dataset, *, batch_size=1, shuffle=False, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.keys = np.arange(0, len(self.dataset))
        if shuffle:
            np.random.shuffle(self.keys)
        self.num_batches = len(dataset) // batch_size
        if not drop_last and self.num_batches * batch_size < len(self.dataset):
            self.num_batches += 1

    def __iter__(self):
        """initializes the iterator, return self as an iterator object"""
        self.batch_num = 0
        return self

    def __next__(self):
        """method which actually performs the iteration"""
        # keep iterator from return nothing forever when it runs out
        # of data
        if self.batch_num >= self.num_batches:
            raise StopIteration
        # initialize lists which we'll append values to and convert
        # to ndarrays later
        X = []
        y = []
        keys = self.keys[self.batch_num * self.batch_size:
                         (self.batch_num + 1) * self.batch_size]
        # probably want to remove tqdm, using it currently to both
        # see what's going on, in particular whether drop_last works
        # as intended
        for key in tqdm(keys):
            feats, label = self.dataset[key]
            X.append(feats)
            y.append(label)
        self.batch_num += 1
        return np.array(X), np.array(y)
