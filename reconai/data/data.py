import numpy as np

from box import Box
from pathlib import Path

import torch
from torch.autograd import Variable
import logging

from reconai.utils.kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.models.bcrnn.dnn_io import to_tensor_format, from_tensor_format
from reconai.models.bcrnn.module import Module
import matplotlib.pyplot as plt

from .Batcher import Batcher
from .Volume import Volume


from .dataloader import DataLoader
from .batcher1 import Batcher
from .sequencer import Sequencer

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


def get_dataset_batchers(in_dir: Path, sequence_len: int):
    dl_tra_val = DataLoader(in_dir / 'train')
    dl_tra_val.load(split_regex='.*_(.*)_', filter_regex='sag')
    dl_test = DataLoader(in_dir / 'test')
    dl_test.load(split_regex='.*_(.*)_', filter_regex='sag')

    logging.info("data loaded")
    sequencer_tr_val = Sequencer(dl_tra_val)
    sequencer_test = Sequencer(dl_test)

    kwargs = {'seed': 11, 'seq_len': sequence_len, 'mean_slices_per_mha': 2, 'max_slices_per_mha': 3, 'q': 0.5}
    train_val_sequences = sequencer_tr_val.generate_sequences(**kwargs)
    test_sequences = sequencer_test.generate_sequences(**kwargs)

    logging.info("sequences created")

    tra_val_batcher = Batcher(dl_tra_val)
    for s in train_val_sequences.items():
        tra_val_batcher.append_sequence(s, norm=1961.06)
    for s in train_val_sequences.items():
        tra_val_batcher.append_sequence(s, norm=1961.06, flip='lr')
    # tra_val_batcher.append_sequence(s, norm=1961.06, rotate_degs=list(range(3)))

    logging.info("Train/Validate batcher generated")

    test_batcher = Batcher(dl_test)
    for s in test_sequences.items():
        test_batcher.append_sequence(s, norm=1961.06)
        # tra_val_batcher.append_sequence(s, norm=1961.06, flip='lr')

    return tra_val_batcher, test_batcher


def append_to_file(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float, val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')
