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
##
## Test file to try to use the nnUNet predictor. Lot slower than the command, so not continued with this.
##

## Some of the helper functions are used in other scripts.

def get_imgs_and_filenames(mhas: Path):
    filenames = []
    images = []

    for file in mhas.iterdir():
        volumes = []
        img = sitk.ReadImage(file)
        imgarr = sitk.GetArrayFromImage(img).astype('float64')
        # z = imgarr.shape[0]
        sequence = []
        for i in range(5):
            # sequence.append(crop_or_pad(imgarr[z // 2, :, :] / 1961.06, (256, 256)))
            sequence.append(crop_or_pad(imgarr[i, :, :] / 1961.06, (256, 256)))
        volumes.append(sequence)
        images.append(np.stack(volumes))
        filenames.append(file)

    return filenames, images


def get_batcher(equal_images: bool = False):
    dl = DataLoader(Path('../../../data_500/test'))
    dl.load(split_regex='.*_(.*)_', filter_regex='sag')
    sequencer = SequenceBuilder(dl)
    kwargs = {'seed': 11, 'seq_len': 5, 'random_order': False}
    test_sequences = sequencer.generate_sequences(**kwargs)
    batcher = Batcher(dl)
    i = 0
    for s in test_sequences.items():
        if i > 9:
            batcher.append_sequence(sequence=s,
                                    crop_expand_to=(256, 256),
                                    norm=1961.06,
                                    equal_images=equal_images,
                                    expand_to_n=equal_images)
        if i > 12:
            return batcher
        i += 1
    return batcher


def get_image(filename):
    volumes = []
    ifr = sitk.ImageFileReader()
    ifr.SetFileName(filename)
    img = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')
    z = img.shape[0]
    sequence = []
    for i in range(5):
        sequence.append(crop_or_pad(img[z//2, :, :] / 1961.06, (256, 256)))
    volumes.append(sequence)
    return np.stack(volumes)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean((2 * intersect) / total_sum)
    return round(dice, 3)


def main():
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
    print_images = sys.argv[2].lower() == 'true'
    filename = f'../../../segmentation/test/{input_filename}'

    batcher = get_batcher(equal_images='equal' in input_filename)

    und = int(filename.split('_')[2])
    filters = int(filename.split('_')[3])
    iters = int(filename.split('_')[4][0])
    print(f'und {und}, filters {filters}, iters {iters}, print {print_images}')

    if print_images:
        save_dir: Path = Path(f'{datetime.now().strftime("%Y%m%d_%H%M")}')
        save_dir.mkdir(parents=True)

    network = CRNNMRI(n_ch=1, nc=iters, nf=filters, ks=3, nd=5).cuda()
    state_dict = torch.load(filename)
    network.load_state_dict(state_dict)
    network.eval()

    with torch.no_grad():

        dices = []

        for image in batcher.items():
            i = 0
            im_u, k_u, mask, gnd = preprocess_as_variable(image, 11, und, equal_mask='equal' in input_filename)

            rec, iters = network(im_u, k_u, mask, False)

            gnd = gnd[0].permute(0, 3, 1, 2).detach().cpu().numpy()
            rec = rec[0].permute(0, 3, 1, 2).detach().cpu().numpy()

            img, props = SimpleITKIO().read_images(['6_4_0000.nii.gz'])
            img2, props2 = SimpleITKIO().read_images(['5_3_0000.nii.gz'])

            ret = predictor.predict_from_list_of_npy_arrays([gnd, rec],
                                                            None,
                                                            # [{'spacing': [3.0, 1.093999981880188, 1.093999981880188]},
                                                            #  {'spacing': [3.0, 1.093999981880188, 1.093999981880188]}],
                                                            [{'spacing': [1.0, 1.0, 1.0]},
                                                             {'spacing': [1.0, 1.0, 1.0]}],
                                                            None, 5, save_probabilities=False)
                                                            # num_processes_segmentation_export=1)
            seg_gnd = np.array(ret[0][2])
            seg_rec = np.array(ret[1][2])
            if np.sum(seg_gnd) > 0 and np.sum(seg_rec) > 0:
                dice_image = dice_coef(seg_gnd, seg_rec)
                dices.append(dice_image)
                print(f'DICE: {dice_image}')
            else:
                print('no needle found')

            # if print_images and i % 5 == 0:
            if dice_image < 0.1:
                plt.imshow(np.abs(im_u[0, 0, :, :, 4].detach().cpu().numpy() / 1961.06),
                           cmap="Greys_r", interpolation="nearest", aspect='auto')
                plt.savefig(f'{save_dir}/{i}_und.png')
                plt.clf()

                plt.imshow(np.abs(gnd[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
                plt.savefig(f'{save_dir}/{i}_gnd.png')
                plt.clf()

                plt.imshow(np.abs(rec[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
                plt.savefig(f'{save_dir}/{i}_rec.png')
                plt.clf()

                plt.imshow(seg_gnd)
                plt.savefig(f'{save_dir}/{i}_seg_gnd.png')
                plt.clf()

                plt.imshow(seg_rec)
                plt.savefig(f'{save_dir}/{i}_seg_rec.png')
                plt.clf()

                result = seg_gnd.astype('float')
                result2 = seg_rec.astype('float')
                result[result == 0] = np.nan
                result2[result2 == 0] = np.nan

                plt.imshow(np.abs(gnd[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
                plt.imshow(result, cmap='jet', alpha=0.5, aspect='auto')
                plt.savefig(f'{save_dir}/{i}_fullseg_gnd.png')
                plt.clf()

                plt.imshow(np.abs(rec[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
                plt.imshow(result2, cmap='jet', alpha=0.5, aspect='auto')
                plt.savefig(f'{save_dir}/{i}_fullseg_rec.png')
                plt.clf()
            i += 1

        print(f'Mean DICE {np.mean(dices)}')
        print(f'STD {np.std(dices)}')


if __name__ == '__main__':
    main()
