import numpy as np
import random
from typing import List
from box import Box
from pathlib import Path

import torch
import logging

from reconai.utils.kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.cascadenet_pytorch.dnn_io import to_tensor_format, from_tensor_format
import matplotlib.pyplot as plt

from .Batcher import Batcher
from .Volume import Volume

def prepare_input(image: np.ndarray, acceleration: float = 4.0) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    image: ndarray - input image of shape (batch_size, n_channels, width, height)
    acceleration: float - controls the undersampling rate. higher the value, more undersampling

    Returns
    ------
    im_und_l: Tensor - undersampled image in image space
    k_und_l: Tensor - undersampled image in K-space
    mask_l: Tensor - undersampling mask in fourier domain (which lines in k-space to keep / which to ignore)
    im_gnd_l: Tensor - ground truth image in image space
    """
    b, s, y, x = image.shape
    mask = np.zeros(image.shape)
    for b_ in range(b):
        for s_ in range(s):
            mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / acceleration, 1 / 3)

    im_und, k_und = cs.undersample(image, mask, centred=True, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(image))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und, complex=True))
    mask_l = torch.from_numpy(to_tensor_format(mask))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def gather_data(data_dir: Path, debug: bool = False):
    data = []
    for patient_dir in data_dir.iterdir():
        try:
            if patient_dir.is_dir():
                files = list(patient_dir.iterdir())
                study_ids = {fn.name.split('_')[1] for fn in files if not fn.name.startswith('tmp')}
                for study_id in study_ids:
                    data.append(Volume(study_id, [fn for fn in files if study_id in fn.name]))
        except:
            continue
        if debug and len(data) > 20:
            break
    return data


def get_data_volumes(args: Box) -> List[Volume]:
    Batcher.batch_size = args.batch_size
    Volume.sequence_length = args.sequence_len

    data = gather_data(args.in_dir, args.debug)
    data_error = Batcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    data_n = len(data)
    logging.info(f"{data_n} volumes found, {len(data_error)} dropped out")
    if data_n < 3:
        raise ValueError('insufficient data for training')

    return data


def get_dataset_batchers(args: Box, data_volumes: List[Volume], n_folds: int, fold: int) -> (Batcher, Batcher, Batcher):
    data_n = list(range(len(data_volumes)))
    # random.seed(args.seed)
    random.seed(5)
    random.shuffle(data_n)
    data_split = np.array_split(data_n, n_folds + 1 if n_folds > 2 else 5)

    k_validation = {fold + 1}
    k_training = set(range(1, len(data_split))).difference(k_validation)
    # k_test is 0

    train = Batcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_training])])
    validate = Batcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_validation])])
    test = Batcher([data_volumes[i] for i in data_split[0]])

    return train, validate, test


def show_images(rec, gnd):
    # a = 1
    # ifr = sitk.ImageFileReader()
    # ifr.SetFileName('../data/10105/10105_19104511215149791073699219583840411343_needle_0.mha')
    # image = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')

    # rec = from_tensor_format(rec.detach().cpu(), True)[0]
    gnd2 = from_tensor_format(gnd.detach().cpu(), True).numpy()[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rec)
    ax1.set_title('Reconstruction')
    ax2.imshow(gnd2[0])
    ax2.set_title('Ground truth')
    ax3.imshow(gnd2[0] - rec)
    ax3.set_title('Difference')
    plt.show()
