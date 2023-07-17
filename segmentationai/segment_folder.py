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
from reconsegcombi import dice_coef


def main():
    # Predictor is slow. Use command
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    model_folder = '../../../segmentation/nnUNet_results/' \
                   'Dataset502_fastmri_intervention/nnUNetTrainer__nnUNetPlans__3d_fullres'
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))

    input_filename = sys.argv[1]
    filename = f'../../../segmentation/test/{input_filename}'

    und = int(filename.split('_')[2])
    filters = int(filename.split('_')[3])
    iters = int(filename.split('_')[4][0])
    print(f'und {und}, filters {filters}, iters {iters}')

    gnds = Path('../../../data_500/test/1')
    recons = Path(f'../../../segmentation/test/recon_ne_{und}_{filters}_{iters}')

    dices = []

    for image in gnds.iterdir():
        gnd = image
        rec = str(gnd).split('/')[-1]
        img, props = SimpleITKIO().read_images([str(gnd)])
        img2, props2 = SimpleITKIO().read_images([str(recons / rec)])

        ret = predictor.predict_from_list_of_npy_arrays([img, img2],
                                                        None,
                                                        [props, props2],
                                                        None, 1, save_probabilities=True,
                                                        num_processes_segmentation_export=1)
        seg_gnd = np.array(ret[0][0][2])
        seg_rec = np.array(ret[1][0][2])

        if np.sum(seg_gnd) > 0 and np.sum(seg_rec) > 0:
            dice_image = dice_coef(seg_gnd, seg_rec)
            dices.append(dice_image)
            print(f'DICE: {dice_image}')
        else:
            print('no needle found')

if __name__ == '__main__':
    main()
