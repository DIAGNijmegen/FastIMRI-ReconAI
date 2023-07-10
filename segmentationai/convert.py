
import SimpleITK as sitk
from pathlib import Path
import os
import numpy as np

from segmentationai.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def convert_dataset_files():

    basepath = Path('../../../segmentation/nnUNet_raw/Dataset502_fastmri_intervention')

    mhas = basepath / 'images-mha'
    niigz = basepath / 'imagesTr'

    # Convert file to .nii.gz and normalize
    for file in mhas.iterdir():

        filename = str(file).split('/')[-1]
        # filename = filename.split('_')
        # filename = '_'.join(filename[:-2]) + f'_{filename[-1]}'
        filename = filename.split('.')
        filename[-2] = filename[-2] + '_0000'
        filename[-1] = 'nii.gz'
        filename = '.'.join(filename)
        img = sitk.ReadImage(file)
        imgarr = sitk.GetArrayFromImage(img)
        imgarr = imgarr / 1961.06
        print(np.mean(imgarr))

        img = sitk.GetImageFromArray(imgarr)
        sitk.WriteImage(img, niigz / filename)

    # Generate JSON
    # generate_dataset_json(basepath, { "0": "needle"}, {"background" : 0,"needle": 1}, 199, '.nii.gz', dataset_name='Dataset502_fastmri_intervention')

    # Find the missing labels / files without label
    # files_in_dir = os.listdir(basepath / 'labelsTr')
    # files_dir = '\t'.join(files_in_dir)
    # for file in mhas.iterdir():
    #     splitted = str(file).split('/')
    #     splittedLast = splitted[-1].split('.')
    #     splittedLast[-1] = ''
    #
    #     if '.'.join(splittedLast) not in files_dir:
    #         print(splittedLast)

if __name__ == '__main__':
    convert_dataset_files()
