import os
import shutil



folders = os.listdir('data/nii/AD/')
for folder in folders:
    fs = os.listdir(f'data/nii/AD/{folder}/MPRAGE')
    if len(fs) > 1:
        for p in fs[:-1]:
            shutil.rmtree(os.path.join('data', 'nii', 'AD', folder, 'MPRAGE', p))

