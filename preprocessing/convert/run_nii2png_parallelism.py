import multiprocessing as mp
import os
import numpy as np
from utils import nii2png
import nibabel as nib
from tqdm import tqdm
import re


def run_nii2png(input_path, output_path, crop):

    filepaths = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.endswith(".nii"):
                filepaths.append(os.path.join(root, f))

    pbar = tqdm(filepaths)
    for filepath in pbar:
        pbar.set_description("Executing nii2png")
        img = nib.load(filepath).get_fdata()

        patient_id = re.search("(?<=ADNI_).{1,10}", filepath).group()
        exam_type = re.search(".{6}(?=_br)", filepath).group()
        exam_id = filepath.split(os.path.sep)[-1]
        exam_id = exam_id.split("_")[-1].replace(".nii", "")

        filename = " ".join([patient_id, exam_type, exam_id])

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        slices = np.arange(80, 111).tolist()

        try:
            # Setup a list of processes that we want to run
            processes = [mp.Process(target=nii2png, args=(img, slices, crop, filename, output_path)) for x in range(6)]

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

        except:
            pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", help="Root path with nii files. First level must have {AD, MCI, CN} folders", required=True)
    parser.add_argument("--output", help="Output path", required=True)
    parser.add_argument("--crop", help="Crop final image", action="store_true")
    args = parser.parse_args()

    run_nii2png(args.input, args.output, args.crop)
