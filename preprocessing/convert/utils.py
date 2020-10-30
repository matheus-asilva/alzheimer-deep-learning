import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc


def nii2png(file, slices=[], crop=False, exam_plane="coronal", filename="IMAGE", output_path=None):
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
    
    # if file.shape != (160, 192, 192):
        # raise ValueError("Different shape. Must be: (160,192,192,1)")

    sizes = {"sagital": file.shape[0], "coronal": file.shape[1], "axial": file.shape[2]}

    if len(slices) == 0:
        slices = np.arange(sizes[exam_plane]).tolist()

    if type(slices) != list:
        slices = [slices]

    for slice_ in slices:
        # fig, ax = plt.subplots(figsize=(0.72, 0.72), dpi=300)
        fig, ax = plt.subplots(dpi=300)

        if exam_plane.lower() == "sagital":
            ax.imshow(file[slice_, :, :], cmap='bone', interpolation='lanczos')
        elif exam_plane.lower() == "coronal":
            if crop:
                ax.imshow(np.rot90(file[:, slice_, :])[65:135, 40:115], cmap="bone", interpolation='lanczos')
            else:
                ax.imshow(np.rot90(file[:, slice_, :]), cmap='bone', interpolation='lanczos')

        elif exam_plane.lower() == "axial":
            ax.imshow(file[:, :, slice_], cmap='bone', interpolation='lanczos')

        ax.axis('off')
        ax.axis('tight')
        ax.axis('image')

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "converted")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.savefig(os.path.join(output_path, ' '.join([filename, f'SLICE_{slice_}.png'])))
        plt.close(fig)
        gc.collect()
