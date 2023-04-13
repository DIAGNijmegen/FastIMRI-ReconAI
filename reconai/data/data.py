import numpy as np
import random
from typing import List
from box import Box
from pathlib import Path

import torch
from torch.autograd import Variable
import logging

from reconai.utils.kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.cascadenet_pytorch.dnn_io import to_tensor_format, from_tensor_format
from reconai.cascadenet_pytorch.module import Module
import matplotlib.pyplot as plt

from .Batcher import Batcher as OldBatcher
from .Volume import Volume
from .dataloader import DataLoader
from .batcher1 import Batcher

def prepare_input_as_variable(image: np.ndarray, acceleration: float = 4.0) \
        -> (torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor):
    im_und, k_und, mask, im_gnd = prepare_input(image, acceleration)
    im_u = Variable(im_und.type(Module.TensorType))
    k_u = Variable(k_und.type(Module.TensorType))
    mask = Variable(mask.type(Module.TensorType))
    gnd = Variable(im_gnd.type(Module.TensorType))

    return im_u, k_u, mask, gnd


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
    OldBatcher.batch_size = args.batch_size
    Volume.sequence_length = args.sequence_len

    data = gather_data(args.in_dir, args.debug)
    data_error = OldBatcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    data_n = len(data)
    logging.info(f"{data_n} volumes found, {len(data_error)} dropped out")
    if data_n < 3:
        raise ValueError('insufficient data for training')

    return data


def get_dataset_batchers(args: Box, data_volumes: List[Volume], n_folds: int, fold: int) -> (OldBatcher, OldBatcher, OldBatcher):
    data_n = list(range(len(data_volumes)))
    # random.seed(args.seed)
    random.seed(5)
    random.shuffle(data_n)
    data_split = np.array_split(data_n, n_folds + 1 if n_folds > 2 else 5)

    k_validation = {fold + 1}
    k_training = set(range(1, len(data_split))).difference(k_validation)
    # k_test is 0

    train = OldBatcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_training])])
    validate = OldBatcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_validation])])
    test = OldBatcher([data_volumes[i] for i in data_split[0]])

    return train, validate, test

def get_new_dataset_batchers(args: Box):
    dl_tr_val = DataLoader(args.in_dir / 'train')
    dl_tr_val.load('.*_(.*)_')
    kwargs = {'seed': 11, 'seq_len': 3, 'mean_slices_per_mha': 2, 'max_slices_per_mha': 3, 'q': 0.5}
    train_val_sequences = dl_tr_val.generate_sequences(**kwargs)
    dl_test = DataLoader(args.in_dir / 'test')
    dl_test.load('.*_(.*)_')
    test_sequences = dl_test.generate_sequences(**kwargs)

    tra_val_batcher = Batcher(dl_tr_val)
    for s in train_val_sequences.items():
        tra_val_batcher.append_sequence(s, norm=1961.06)

    test_batcher = Batcher(dl_test)
    for s in test_sequences.items():
        test_batcher.append_sequence(s, norm=1961.06)

    return tra_val_batcher, test_batcher


def append_to_file(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float, val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')


def show_images(rec, gnd):
    # a = 1
    # ifr = sitk.ImageFileReader()
    # ifr.SetFileName('../data/10105/10105_19104511215149791073699219583840411343_sag_0.mha')
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


def get_data_information(args):
    # Volume.key = 'needle'
    data = get_data_volumes(args)
    train, validate, _ = get_dataset_batchers(args, data, 1, 0)

    mins = []
    maxes = []
    max_to_perc99 = []
    averages = []
    for image in train.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            mins.append(slice.min())
            maxes.append(slice.max())
            averages.append(slice.mean())
            perc99 = np.percentile(slice, 99)
            max_to_perc99.append(slice.max() - perc99)

    for image in validate.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            maxes.append(slice.max())

    # print(f'Min {min(mins)}')
    print(f'Min of maxes {min(maxes)}')
    print(f'Average of maxes {np.mean(maxes)}')
    print(f'std of maxes {np.std(maxes)}')
    print(f'Max of maxes {max(maxes)}')
    # print(f'Avg {np.mean(averages)}')
    # print(f'Min diff max-perc99 {np.min(max_to_perc99)}')
    # print(f'Avg diff max-perc99 {np.mean(max_to_perc99)}')
    # print(f'Max diff max-perc99 {np.max(max_to_perc99)}')
