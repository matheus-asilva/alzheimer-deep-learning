import os
import numpy as np
import nibabel as nib
import ants
import time
from tqdm import tqdm


#registration
def registration(fixed_image, moving_image, type_of_transform='SyN'):
    return ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform=type_of_transform)['warpedmovout']


#noise reduction
def noise_reduction(img):
    return ants.denoise_image(img)


#bias field correction
def bias_field_correction(img):
    return ants.n3_bias_field_correction(img)

def small_main(img_path):
    img = ants.image_read(img_path)
    img_corrected = bias_field_correction(img)

    type_folder = img_path.split(os.path.sep)[3]
    output_path = os.path.join('data', 'nii', 'processed', type_folder)
    filename = img_path.split(os.path.sep)[-1]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nii_target = ants.to_nibabel(img_corrected)
    nib.save(nii_target, os.path.join(output_path, filename))


def full_main(fixed, target):
    fixed_img = ants.image_read(fixed)
    target_img = ants.image_read(target)

    target_corrected = registration(fixed_image=fixed_img, moving_image=target_img)
    target_corrected = noise_reduction(target_corrected)
    target_corrected = bias_field_correction(target_corrected)
    
    type_folder = target.split(os.path.sep)[3]
    output_path = os.path.join('data', 'nii', 'processed', type_folder)
    filename = target.split(os.path.sep)[-1]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nii_target = ants.to_nibabel(target_corrected)
    nib.save(nii_target, os.path.join(output_path, filename))

if __name__ == "__main__":
    
    paths = []
    for root, _, files in os.walk("data/nii/raw"):
        for f in files:
            paths.append(os.path.join(root, f))
    
    pbar = tqdm(paths)
    for target_image in pbar:
        if target_image.split(os.path.sep)[3] == "AD":
            fixed_image = "data/nii/raw/AD/003_S_1059/MPRAGE/2008-12-23_09_15_12.0/S61129/ADNI_003_S_1059_MR_MPRAGE_br_raw_20081224065743160_102_S61129_I132187.nii"
        elif target_image.split(os.path.sep)[3] == "CN":
            fixed_image = "data/nii/raw/CN/003_S_0907/MPRAGE/2007-10-30_10_13_36.0/S42056/ADNI_003_S_0907_MR_MPRAGE_br_raw_20071030140717906_58_S42056_I79751.nii"

        pbar.set_description("Normalizing nii files")
        small_main(img_path=target_image)
        # full_main(fixed=fixed_image, target=target_image)


    # #plot
    # ants.plot(img2_corrected, axis=1)
    # ants.plot(img2, axis=2)
