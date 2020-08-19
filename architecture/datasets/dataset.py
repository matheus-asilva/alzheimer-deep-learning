from pathlib import Path
import os
from architecture import util

class Dataset:

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / 'data'
    
    def load_or_generate_data(self):
        pass

def _download_raw_dataset(metadata):
    if os.path.exists(metadata['filename']):
        return
    print('Downloading raw dataset from %s...' % metadata['url'])
    util.download_url(metadata['url'], metadata['filename'])