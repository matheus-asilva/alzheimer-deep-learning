import os
import numpy as np
from utils import nii2png
import nibabel as nib
from tqdm import tqdm

def run_nii2png(input_path):

    filepaths = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.endswith(".nii"):
                filepaths.append(os.path.join(root, f))

    pbar = tqdm(filepaths)
    for filepath in pbar:
        pbar.set_description("Executing nii2png")
        img = nib.load(filepaths[0]).get_fdata()

        patient_id = filepath.split(os.path.sep)[-5]
        exam_type = filepath.split(os.path.sep)[-4]
        exam_id = filepath.split(os.path.sep)[-1].split("_")[-1].replace(".nii", "")
        disease = filepath.split(os.path.sep)[-6]

        filename = " ".join([patient_id, exam_type, exam_id])

        output_path = os.path.join(input_path.replace("raw", os.path.join("converted")), disease)
        slices = np.arange(105, 126).tolist()
        nii2png(img, slices=slices, filename=filename, output_path=output_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", help="Root path with nii files. First level must have {AD, MCI, CN} folders", required=True)
    args = parser.parse_args()
    run_nii2png(args.input)
