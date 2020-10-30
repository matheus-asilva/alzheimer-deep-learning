import os
import numpy as np
from tqdm import tqdm
import re
import random


def get_unique_ids_and_paths(path):
    filepath = []
    ids = []
    for root, _, files in os.walk(path):
        pbar = tqdm(files)
        for f in pbar:
            pbar.set_description("Getting Unique IDs for patients")
            
            # Mapping png files
            fullpath = os.path.join(root, f)
            filepath.append(fullpath)

            # Mapping patient id
            dir = root.split(os.path.sep)[-1]

            if os.path.sep == "\\":
                sep = os.path.sep + "\\"
            else:
                sep = os.path.sep

            ids.append(re.search(f'(?<={dir}{sep})(.*)(?= MPRAGE)', fullpath.upper()).group(1))

    return list(np.unique(ids)), filepath


def sampling(ids: list, seed: int = 42):
    random.seed(seed)

    random.shuffle(ids)

    train_index = int(np.round(len(ids) * .7))
    val_index = int(np.round(len(ids) * .2))
    test_index = int(np.round(len(ids) * .1))

    return {
        'train': ids[:train_index],
        'validation': ids[train_index: train_index + val_index],
        'test': ids[-test_index:]
        }
        