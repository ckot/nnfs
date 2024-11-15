from pathlib import Path
import os
import shutil

from tqdm import tqdm

ORIG_DATA = Path("fashion_mnist_images")
ORIG_TRAIN_DATA = ORIG_DATA / Path("train")
ORIG_TEST_DATA = ORIG_DATA / Path("test")

NEW_DATA = Path("fashion_mnist")

NEW_TRAIN_DATA = NEW_DATA /Path("train")
NEW_TRAIN_DATA_IMAGES = NEW_TRAIN_DATA / Path("images")
NEW_TRAIN_DATA_LABELS_FILE = NEW_TRAIN_DATA / Path('labels.csv')

NEW_TEST_DATA = NEW_DATA / Path("test")
NEW_TEST_DATA_IMAGES = NEW_TEST_DATA / Path("images")
NEW_TEST_DATA_LABELS_FILE = NEW_TEST_DATA / Path("labels.csv")

NEW_TRAIN_DATA_IMAGES.mkdir(parents=True, exist_ok=True)
NEW_TEST_DATA_IMAGES.mkdir(parents=True, exist_ok=True)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def transform_filename(label_num, orig_filename):
    prefix = labels_map[int(label_num)]
    prefix = prefix.lower().replace('-', '').replace(' ', '')
    new_file_name = prefix + "_" + orig_filename
    return new_file_name

def copy_files(orig_data_path, new_data_path, new_labels_file):
    print(f"copying files to {new_data_path}...")
    with open(new_labels_file, "w") as csv:

        for label_num in tqdm(sorted(os.listdir(orig_data_path))):
            # print(label_num)
            for filename in tqdm(sorted(os.listdir(orig_data_path / Path(label_num)))):
                new_filename = transform_filename(label_num, filename)
                # print(f"cp {orig_data_path}/{label_num}/{filename} {new_data_path}/{new_filename}")
                # shutil.copyfile(orig_data_path / Path(label_num) / Path(filename),
                #                 new_data_path / Path(new_filename))
                # print(f"append to {new_labels_file}: {new_filename}, {label_num}")
                csv.write(f'"{new_filename}", {label_num}\n')
                # break

copy_files(ORIG_TRAIN_DATA, NEW_TRAIN_DATA_IMAGES, NEW_TRAIN_DATA_LABELS_FILE)
copy_files(ORIG_TEST_DATA,  NEW_TEST_DATA_IMAGES,  NEW_TEST_DATA_LABELS_FILE)