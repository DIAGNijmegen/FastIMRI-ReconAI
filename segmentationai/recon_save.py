import torch
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import matplotlib.pyplot as plt
import numpy as np
from reconai.model.model_pytorch import CRNNMRI
import SimpleITK as sitk
from reconai.data.data import preprocess_as_variable
from reconai.data.batcher import crop_or_pad
import sys
from datetime import datetime
from pathlib import Path

from reconai.data.data import DataLoader
from reconai.data.sequencebuilder import SequenceBuilder
from reconai.data.batcher import Batcher
from reconsegcombi import get_imgs_and_filenames


def main():
    # input_filename = sys.argv[1]
    filenames, images = get_imgs_and_filenames(Path('../../../segmentation/test/gnd_mhas'))

    list_filenames = [
        'crnn_ne_8_64_5.npz', 'crnn_ne_16_64_5.npz', 'crnn_ne_25_64_5.npz', 'crnn_ne_32_64_5.npz',
        # 'bcrnn_equal_8_64_5.npz', 'bcrnn_equal_16_64_5.npz', 'bcrnn_equal_25_64_5.npz', 'bcrnn_equal_32_64_5.npz',
        'crnn_ne_8_128_10.npz', 'crnn_ne_16_128_10.npz', 'crnn_ne_25_128_10.npz', 'crnn_ne_32_128_10.npz',
        # 'bcrnn_equal_8_128_10.npz', 'bcrnn_equal_16_128_10.npz', 'bcrnn_equal_25_128_10.npz', 'bcrnn_equal_32_128_10.npz'
    ]

    for npzfilename in list_filenames:
        filename = f'../../../segmentation/test/{npzfilename}'

        mask = filename.split('_')[1]
        und = int(filename.split('_')[2])
        filters = int(filename.split('_')[3])
        iters = int(filename.split('_')[4][0])
        if iters == 1:
            iters = 10
        print(f'und {und}, filters {filters}, iters {iters}')

        save_dir: Path = Path(f'../../../segmentation/test/recon_crnn_{mask}_{und}_{filters}_{iters}')
        save_dir.mkdir(parents=True, exist_ok=True)

        network = CRNNMRI(n_ch=1, nc=iters, nf=filters, ks=3, nd=5, single_crnn=True).cuda()
        state_dict = torch.load(filename)
        network.load_state_dict(state_dict)
        network.eval()

        with torch.no_grad():
            for i in range(len(images)):
                image = images[i]

                if 'equal' in npzfilename:
                    img0 = get_image_nontemporal(image, 0)
                    im_u_0, k_u_0, mask_0, _ = preprocess_as_variable(img0, 11, und, equal_mask=True)
                    rec0, _ = network(im_u_0, k_u_0, mask_0, False)
                    rec0 = rec0[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]

                    img1 = get_image_nontemporal(image, 1)
                    im_u_1, k_u_1, mask_1, _ = preprocess_as_variable(img1, 12, und, equal_mask=True)
                    rec1, _ = network(im_u_1, k_u_1, mask_1, False)
                    rec1 = rec1[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]

                    img2 = get_image_nontemporal(image, 2)
                    im_u_2, k_u_2, mask_2, _ = preprocess_as_variable(img2, 13, und, equal_mask=True)
                    rec2, _ = network(im_u_2, k_u_2, mask_2, False)
                    rec2 = rec2[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]

                    img3 = get_image_nontemporal(image, 3)
                    im_u_3, k_u_3, mask_3, _ = preprocess_as_variable(img3, 14, und, equal_mask=True)
                    rec3, _ = network(im_u_3, k_u_3, mask_3, False)
                    rec3 = rec3[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]

                    img4 = get_image_nontemporal(image, 4)
                    im_u_4, k_u_4, mask_4, _ = preprocess_as_variable(img4, 15, und, equal_mask=True)
                    rec4, _ = network(im_u_4, k_u_4, mask_4, False)
                    rec4 = rec4[0].permute(0, 3, 1, 2).detach().cpu().numpy()[0]

                    rec0[1] = rec1[1]
                    rec0[2] = rec2[2]
                    rec0[3] = rec3[3]
                    rec0[4] = rec4[4]
                    rec = rec0
                else:
                    im_u, k_u, mask, gnd = preprocess_as_variable(image, 11, und, equal_mask=False)
                    rec, _ = network(im_u, k_u, mask, False)

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

def get_image_nontemporal(image: np.ndarray, index: int):
    for i in range(5):
        image[0][0][i] = image[0][0][index]
    return image


if __name__ == '__main__':
    main()
