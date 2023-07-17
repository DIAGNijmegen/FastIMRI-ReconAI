import torch
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import matplotlib.pyplot as plt
import numpy as np
from reconai.model.model_pytorch import CRNNMRI
import SimpleITK as sitk
from reconai.data.data import prepare_input_as_variable
from reconai.data.batcher import crop_or_pad
import sys
from datetime import datetime
from pathlib import Path

from reconai.data.data import DataLoader
from reconai.data.sequencebuilder import SequenceBuilder
from reconai.data.batcher import Batcher
from reconsegcombi import get_imgs_and_filenames


def main():
    input_filename = sys.argv[1]
    filename = f'../../../segmentation/test/{input_filename}'

    filenames, images = get_imgs_and_filenames()

    mask = filename.split('_')[1]
    und = int(filename.split('_')[2])
    filters = int(filename.split('_')[3])
    iters = int(filename.split('_')[4][0])
    print(f'und {und}, filters {filters}, iters {iters}')

    save_dir: Path = Path(f'../../../segmentation/test/recon_{mask}_{und}_{filters}_{iters}')
    save_dir.mkdir(parents=True, exist_ok=True)

    network = CRNNMRI(n_ch=1, nc=iters, nf=filters, ks=3, nd=5).cuda()
    state_dict = torch.load(filename)
    network.load_state_dict(state_dict)
    network.eval()

    with torch.no_grad():
        for i in range(len(images)):
            image = images[i]
            im_u, k_u, mask, gnd = prepare_input_as_variable(image, 11, und, equal_mask='equal' in input_filename)

            rec, iters = network(im_u, k_u, mask, False)

            rec = rec[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]
            filename = str(filenames[i]).split('/')[-1]

            filename = filename.split('.')
            if '0000' not in filename:
                filename[-2] = filename[-2] + '_0000'
            filename[-1] = 'nii.gz'
            filename = '.'.join(filename)

            flnm = Path(save_dir / filename)
            rec_img = sitk.GetImageFromArray(rec)
            sitk.WriteImage(rec_img, flnm)

if __name__ == '__main__':
    main()
