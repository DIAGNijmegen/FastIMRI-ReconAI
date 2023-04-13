from reconai.data.data import get_data_volumes, get_dataset_batchers, prepare_input_as_variable
import torch
from reconai.models.bcrnn.model_pytorch import CRNNMRI
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from reconai.models.bcrnn.dnn_io import from_tensor_format
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from reconai.utils.kspace import kspace_to_image, image_to_kspace
from reconai.data.Volume import Volume


def main_presentatie_stan():
    accelerations = [1, 32, 64]  # [1, 2, 4, 8, 12, 16, 32, 64]
    image = get_image()
    for acceleration in accelerations:
        filename = f'../data/model_checkpoints/acc_{acceleration}.npz'

        network = CRNNMRI(n_ch=1, nc=5).cuda()
        state_dict = torch.load(filename)
        network.load_state_dict(state_dict)
        network.eval()

        im_u, k_u, mask, gnd = prepare_input_as_variable(image, acceleration)

        pred = network(im_u, k_u, mask, test=True)
        pred = from_tensor_format(pred.detach().cpu(), True)
        fig, ax = plt.subplots()
        ax.imshow(np.abs(pred.squeeze()[14]), cmap="Greys_r")
        plt.show()
        plt.close(fig)
        # plt.savefig(f'../data/stan_pres/acc_{acceleration}.png')

        im_u = from_tensor_format(im_u.detach().cpu(), True)
        fig, ax = plt.subplots()
        plt.imshow(np.abs(im_u.squeeze()[14]), cmap="Greys_r")
        plt.show()
        plt.close(fig)
        # plt.savefig(f'../data/stan_pres/acc_{acceleration}_und.png')


def get_image():
    volumes = []
    ifr = sitk.ImageFileReader()
    ifr.SetFileName('../data/stan_pres/10105.mha')
    img = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')
    z = img.shape[0]
    sequence = []
    for i in range(15):
        sequence.append(img[z//2, :, :] / 1961.06)
    volumes.append(sequence)
    return np.stack(volumes)


def normalization_test(args):
    def show_im(original, minmaxscaled, standardscaled, sampledimageminmax, sampledimagestandard):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(original)
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(minmaxscaled)
        axs[0, 1].set_title('MinMaxScaled')
        axs[0, 2].imshow(sampledimageminmax)
        axs[0, 2].set_title('Sampled')
        axs[0, 3].imshow(sampledimageminmax - minmaxscaled)
        axs[0, 3].set_title('MM - S diff')
        axs[1, 0].imshow(original)
        axs[1, 0].set_title('Original Image')
        axs[1, 1].imshow(standardscaled)
        axs[1, 1].set_title('StandardScaled')
        axs[1, 2].imshow(sampledimagestandard)
        axs[1, 2].set_title('Sampled')
        axs[1, 3].imshow(sampledimagestandard - standardscaled)
        axs[1, 3].set_title('SS - S diff')
        plt.show()

    Volume.key = 'needle'
    data = get_data_volumes(args)
    train, _, _ = get_dataset_batchers(args, data, 1, 0)

    image = next(train.generate())

    scaler = MinMaxScaler()
    minmax_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        minmax_scaled_image[0, layer, :, :] = scaler.fit_transform(minmax_scaled_image[0, layer, :, :])

    scaler = StandardScaler()
    standard_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        standard_scaled_image[0, layer, :, :] = scaler.fit_transform(standard_scaled_image[0, layer, :, :])

    sampled_mm = kspace_to_image(image_to_kspace(minmax_scaled_image[0][0]))
    sampled_ss = kspace_to_image(image_to_kspace(standard_scaled_image[0][0]))

    show_im(image[0][0], minmax_scaled_image[0][0], standard_scaled_image[0][0], sampled_mm, sampled_ss)


if __name__ == '__main__':
    main_presentatie_stan()
