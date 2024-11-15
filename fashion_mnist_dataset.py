
import numpy as np
import os
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
