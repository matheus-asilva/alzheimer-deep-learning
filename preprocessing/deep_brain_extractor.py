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

def run_deepbrain(paths):
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
        finalpath = os.path.join('finished', fullpath)

        if not os.path.exists(finalpath):
            os.makedirs(finalpath)
        
        os.system("deepbrain-extractor -i {main_file} -o {destiny}".format(main_file=f, destiny=finalpath))

        for nii in os.listdir(finalpath):
            if nii.endswith(".nii"):
                os.rename(os.path.join(finalpath, nii), os.path.join(finalpath, nii).replace("brain", filename))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help=r"Path which contains types of disease. Must contain {AD, MCI, CN} folders", required=True)
    args = parser.parse_args()

    paths = get_paths(args.path)
    # TODO
    paths[0]
