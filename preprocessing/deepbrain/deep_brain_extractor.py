import os
from tqdm import tqdm

def get_paths(path):
    """Function to get the path of each nifti file.

    Args:
        path (str): Folder which contains disease folders with each nifti file

    Returns:
        list of paths
    """

    paths = []
    for root, _, files in os.walk(path):
        for f in files:
            paths.append(os.path.join(root, f))

    return paths

def run_deepbrain(paths, output_path):
    """Function to execute deepbrain-extractor for each nifti file. This method extracts all except the brain

    Args:
        paths (list): List of paths for the nifti files

    Returns:
        None
    """

    for f in tqdm(paths):
        f = os.path.normpath(f)
        filename = os.path.basename(f).split(".nii")[0]
        fullpath = os.path.dirname(f)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        os.system("deepbrain-extractor -i {main_file} -o {output_path}".format(main_file=f, output_path=output_path))

        for nii in os.listdir(output_path):
            if nii.endswith(".nii"):
                os.rename(os.path.join(output_path, nii), os.path.join(output_path, nii).replace("brain", filename))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path which contains types of disease. Must contain {AD, MCI, CN} folders", required=True)
    parser.add_argument("--output_path", help="Output path where files will be saved", required=True)
    args = parser.parse_args()

    paths = get_paths(args.path)
    run_deepbrain(paths=paths, output_path=args.output_path)
