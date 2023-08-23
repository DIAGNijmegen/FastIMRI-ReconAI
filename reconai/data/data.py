import numpy as np
from pathlib import Path

import torch
from torch.autograd import Variable
import logging

from reconai.utils.kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.model.dnn_io import to_tensor_format
from reconai.model.module import Module

from .sequencebuilder import SequenceBuilder, SequenceCollection
from .dataloader import DataLoader
from .batcher import Batcher
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

def get_dataloader(params: Parameters, path_suffix: str) -> DataLoader:
    dl = DataLoader(params.in_dir / path_suffix)
    dl.load(split_regex=params.config.data.split_regex, filter_regex=params.config.data.filter_regex)
    return dl

def generate_sequences(params: Parameters, dl: DataLoader, multislice: bool = True) -> SequenceCollection:
    sequencer = SequenceBuilder(dl)
    if multislice:
        kwargs = {
            'seed': params.config.data.sequence_seed,
            'seq_len': params.config.data.sequence_length,
            'mean_slices_per_mha': params.config.data.mean_slices_per_mha,
            'max_slices_per_mha': params.config.data.max_slices_per_mha,
            'q': params.config.data.q
        }
        return sequencer.generate_multislice_sequences(**kwargs)
    else:
        kwargs = {
            'seed': params.config.data.sequence_seed,
            'seq_len': params.config.data.sequence_length,
            'random_order': False
        }
        return sequencer.generate_singleslice_sequences(**kwargs)

def get_batcher(params: Parameters, dl: DataLoader, sequences: SequenceCollection,
                equal_images: bool = False, expand_to_n: bool = False):
    batcher = Batcher(dl)
    for s in sequences.items():
        batcher.append_sequence(sequence=s,
                                crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x),
                                norm=params.config.data.normalize,
                                equal_images=equal_images,
                                expand_to_n=expand_to_n)
    return batcher

def get_dataset_batchers(params: Parameters):
    dl_tra_val = get_dataloader(params, 'train')
    dl_test = get_dataloader(params, 'test')
    logging.info("data loaded")

    train_val_sequences = generate_sequences(params, dl_tra_val, multislice=params.config.data.multislice)
    test_sequences = generate_sequences(params, dl_test, multislice=params.config.data.multislice)
    logging.info(f"{len(train_val_sequences)} train/val sequences created")
    logging.info(f"{len(test_sequences)} test sequences created")

    tra_val_batcher = get_batcher(params, dl_tra_val, train_val_sequences,
                                  equal_images=params.config.data.equal_images,
                                  expand_to_n=params.config.data.expand_to_n)

    test_batcher_equal = get_batcher(params, dl_test, test_sequences, equal_images=True)
    test_batcher_non_equal = get_batcher(params, dl_test, test_sequences, equal_images=False)

    return tra_val_batcher, test_batcher_equal, test_batcher_non_equal
