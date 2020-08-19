import json
import os
from pathlib import Path
import shutil
import zipfile
from imutils import paths

from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import toml
from tqdm import tqdm
import pandas as pd

from architecture.datasets.dataset import _download_raw_dataset, Dataset
from architecture.util import download_file_from_google_drive

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'alzheimer_t2mri'
RAW_DATA_FILENAME = Dataset.data_dirname() / 'raw' / 'alzheimer_t2mri' / 'alzheimer_t2mri.zip'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'alzheimer_t2mri'

ESSENTIALS_FILENAME = Path(os.path.join(os.path.abspath('.'), 'architecture', 'datasets', 'alzheimer_essentials.json'))
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / 'alzheimer_essentials.json'

class AlzheimerT2Dataset(Dataset):
    def __init__(self):
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = dict(essentials['mapping'])
        self.inverse_mapping = {value: key for key, value in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = tuple(essentials['input_shape'])
        self.output_shape = (self.num_classes, )

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    
    def load_images(self, path):
        data = []
        labels = []
        for image_path in tqdm(list(paths.list_images(path))):
            label = image_path.split(os.path.sep)[-2]

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.input_shape)

            data.append(image)
            labels.append(label)
        
        labels = pd.Series(labels).map(self.inverse_mapping)

        return np.array(data), np.array(labels)
        
    def load_or_generate_data(self):
        if 'alzheimer_t2mri' not in os.listdir(os.path.join('data', 'processed')):
            _download_and_process_alzheimer_t2mri()
        
        print('Reading training images...')
        self.X_train, self.y_train = self.load_images(os.path.join(PROCESSED_DATA_DIRNAME, 'train'))
        
        print('Reading validation images...')
        self.X_val, self.y_val = self.load_images(os.path.join(PROCESSED_DATA_DIRNAME, 'validation'))
    
    @cachedproperty
    def y_train(self):
        return to_categorical(self.y_train, self.num_classes)

    @cachedproperty
    def y_val(self):
        return to_categorical(self.y_val, self.num_classes)
    
    def __repr__(self):
        return f'Alzheimer T2 MRI Dataset\nNum classes: {self.num_classes}\nMapping: {self.mapping}\nInput shape: {self.input_shape}'

def _download_and_process_alzheimer_t2mri():
    print('Downloading dataset...')
    metadata = toml.load(METADATA_FILENAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    download_file_from_google_drive(metadata['file_id'],  metadata['filename'])
    _process_raw_dataset(metadata['filename'])
    os.chdir(curdir)

def _process_raw_dataset(filename: str):
    print('Unzipping raw file...')
    
    if not os.path.exists(PROCESSED_DATA_DIRNAME):
        os.mkdir(PROCESSED_DATA_DIRNAME)

    with zipfile.ZipFile(filename, 'r') as file:
        file.extractall(PROCESSED_DATA_DIRNAME)
        file.close()

def main():
    dataset = AlzheimerT2Dataset()
    dataset.load_or_generate_data()
    
    print(dataset)
    print('Train dataset:', dataset.X_train.shape, dataset.y_train.shape)
    print('Validation dataset:', dataset.X_val.shape, dataset.y_val.shape)


if __name__ == '__main__':
    main()