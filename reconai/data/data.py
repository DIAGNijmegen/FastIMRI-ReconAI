import numpy as np
from pathlib import Path

import torch
from torch.autograd import Variable
import logging

from reconai.utils.kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.model.dnn_io import to_tensor_format
from reconai.model.module import Module

from .dataloader import DataLoader
from .batcher import Batcher
from .sequencebuilder import SequenceBuilder
from reconai.parameters import Parameters


def prepare_input_as_variable(image: np.ndarray, seed: int, acceleration: float = 4.0, equal_mask: bool = False) \
        -> (torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor):
    im_und, k_und, mask, im_gnd = prepare_input(image, seed, acceleration, equal_mask)
    im_u = Variable(im_und.type(Module.TensorType))
    k_u = Variable(k_und.type(Module.TensorType))
    mask = Variable(mask.type(Module.TensorType))
    gnd = Variable(im_gnd.type(Module.TensorType))

    return im_u, k_u, mask, gnd


def prepare_input(image: np.ndarray, seed: int, acceleration: float = 4.0, equal_mask: bool = False) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    image: ndarray - input image of shape (batch_size, n_channels, width, height)
    seed: int - the seed to use for the randomization in the mask
    acceleration: float - controls the undersampling rate. higher the value, more undersampling
    equal_mask: bool - If true then all sequences receive the same undersampling mask

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
            mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / acceleration, 1 / 3, seed if equal_mask else seed + s_)

    im_und, k_und = cs.undersample(image, mask, centred=True, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(image))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und, complex=True))
    mask_l = torch.from_numpy(to_tensor_format(mask))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def get_dataset_batchers(params: Parameters):
    dl_tra_val = DataLoader(params.in_dir / 'train')
    dl_tra_val.load(split_regex=params.config.data.split_regex, filter_regex=params.config.data.filter_regex)
    dl_test = DataLoader(params.in_dir / 'test')
    dl_test.load(split_regex=params.config.data.split_regex, filter_regex=params.config.data.filter_regex)

    logging.info("data loaded")
    sequencer_tr_val = SequenceBuilder(dl_tra_val)
    sequencer_test = SequenceBuilder(dl_test)

    kwargs = {'seed': params.config.data.sequence_seed,
              'seq_len': params.config.data.slices,
              'mean_slices_per_mha': params.config.data.mean_slices_per_mha,
              'max_slices_per_mha': params.config.data.max_slices_per_mha,
              'q': params.config.data.q}
    train_val_sequences = sequencer_tr_val.generate_sequences(**kwargs)
    test_sequences = sequencer_test.generate_sequences(**kwargs)

    logging.info("sequences created")

    tra_val_batcher = Batcher(dl_tra_val)

    for s in train_val_sequences.items():
        tra_val_batcher.append_sequence(sequence=s,
                                        crop_expand_to=params.config.data.shape,
                                        norm=params.config.data.normalize,
                                        equal_images=params.config.data.equal_images)

    test_batcher = Batcher(dl_test)
    for s in test_sequences.items():
        test_batcher.append_sequence(sequence=s,
                                     crop_expand_to=params.config.data.shape,
                                     norm=params.config.data.normalize,
                                     equal_images=params.config.data.equal_images)

    return tra_val_batcher, test_batcher


def append_to_file(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float, val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')
