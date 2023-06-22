import torch
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import matplotlib.pyplot as plt
import numpy as np
from reconai.model.model_pytorch import CRNNMRI
import SimpleITK as sitk
from reconai.data.data import prepare_input_as_variable
from reconai.data.batcher import crop_or_pad


def get_image():
    volumes = []
    ifr = sitk.ImageFileReader()
    ifr.SetFileName('../../../segmentation/test/0.mha')
    img = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')
    z = img.shape[0]
    sequence = []
    for i in range(5):
        sequence.append(crop_or_pad(img[z//2, :, :] / 1961.06, (256, 256)))
    volumes.append(sequence)
    return np.stack(volumes)

def main():
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    model_folder = '../../../segmentation/nnUNet_results/' \
                   'Dataset500_fastmri_intervention/nnUNetTrainer__nnUNetPlans__3d_fullres'

    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))

    filename = '../../../segmentation/test/bcrnn.npz'

    network = CRNNMRI(n_ch=1, nc=5, nf=64, ks=3, nd=5).cuda()
    state_dict = torch.load(filename)
    network.load_state_dict(state_dict)
    network.eval()
    image = get_image()

    im_u, k_u, mask, gnd = prepare_input_as_variable(image, 11, 25, True)

    rec, iters = network(im_u, k_u, mask, gnd)

    gnd = gnd[0].permute(0, 3, 1, 2).detach().cpu().numpy()
    rec = rec[0].permute(0, 3, 1, 2).detach().cpu().numpy()

    ret = predictor.predict_from_list_of_npy_arrays([gnd, rec],
                                                    None,
                                                    [{'spacing': [3.0, 1.093999981880188, 1.093999981880188]},
                                                     {'spacing': [3.0, 1.093999981880188, 1.093999981880188]}],
                                                    None, 1, save_probabilities=False,
                                                    num_processes_segmentation_export=1)

    plt.imshow(np.abs(im_u[0, 0, :, :, 4].detach().cpu().numpy() / 1961.06),
               cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.savefig('und.png')
    plt.clf()

    plt.imshow(np.abs(gnd[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.savefig('gnd.png')
    plt.clf()

    plt.imshow(np.abs(rec[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.savefig('rec.png')
    plt.clf()

    result = np.array(ret[0][2])
    result2 = np.array(ret[1][2])
    print(result.shape)
    print(result2.shape)

    plt.imshow(result)
    plt.savefig('seg_gnd.png')
    plt.clf()

    plt.imshow(result2)
    plt.savefig('seg_rec.png')
    plt.clf()

    result = result.astype('float')
    result2 = result2.astype('float')
    result[result == 0] = np.nan
    result2[result2 == 0] = np.nan

    plt.imshow(np.abs(gnd[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.imshow(result, cmap='jet', alpha=0.5, aspect='auto')
    plt.savefig('fullseg_gnd.png')
    plt.clf()

    plt.imshow(np.abs(rec[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.imshow(result2, cmap='jet', alpha=0.5, aspect='auto')
    plt.savefig('fullseg_rec.png')
    plt.clf()


if __name__ == '__main__':
    main()
