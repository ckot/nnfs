import os
import urllib
import urllib.request

from zipfile import ZipFile

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'
SUB_FOLDER = 'train'
if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}")
    urllib.request.urlretrieve(URL, FILE)
    print('Down')

if not os.path.isdir(os.path.join(FOLDER, SUB_FOLDER)):
    # if the 'fashion_mnist_images/train' doesn't exist, the .zip file
    # hasn't been expanded yet
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print("unzipped")