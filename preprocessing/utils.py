import os
import nibabel
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import re
import random


def nii2png(file, slices=[], exam_plane="coronal", filename="IMAGE", output_path=None):
    """Function to convert nifti file to png for each or selected slices

    Args:
        file (np.memmap): Nifti file
        slices (list): List of slices' number
        exam_plane (str): Type of exam. Must be 'sagital'[0], 'coronal'[1], 'axial'[2]
        filename (str): Name of png file
        output_path (str): Path where files will be saved

    Return:
        None 
    """

    sizes = {"sagital": file.shape[0], "coronal": file.shape[1], "axial": file.shape[2]}

    if len(slices) == 0:
        slices = np.arange(sizes[exam_plane]).tolist()

    if type(slices) != list:
        slices = [slices]

    pbar = tqdm(slices)
    for slice_ in pbar:
        pbar.set_description("Converting .nii file to png")
        fig, ax = plt.subplots(dpi=300)

        if exam_plane.lower() == "sagital":
            ax.imshow(file[slice_, :, :, 0], cmap='bone')
        elif exam_plane.lower() == "coronal":
            ax.imshow(file[:, slice_, :, 0], cmap='bone')
        elif exam_plane.lower() == "axial":
            ax.imshow(file[:, :, slice_, 0], cmap='bone')

        ax.axis('off')
        ax.axis('tight')
        ax.axis('image')

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "converted")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.savefig(os.path.join(output_path, ' '.join([filename, f'SLICE_{slice_ + 1}.png'])))
        plt.close(fig)
        gc.collect()

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


def sampling(ids: list, seed: int=42):
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