import json
import shutil
from pathlib import Path
from typing import List

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

import reconai.math.compressed_sensing as cs
from reconai.model.dnn_io import to_tensor_format
from reconai.model.module import Module
from reconai.math.kspace import get_rand_exp_decay_mask
from reconai import version


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, *, normalize: float = 0, as_float32: bool = True):
        self._data_paths: List[Path] = [item.resolve() for item in data_dir.iterdir() if
                                        item.suffix in ['.npy', '.mha']]
        self._data_len = len(self._data_paths)
        if self._data_len == 0:
            raise ValueError(f'no .mha files found at {data_dir}!')

        self._as_float32 = as_float32
        self._normalize = normalize

    @property
    def normalize(self) -> float:
        return self._normalize

    @normalize.setter
    def normalize(self, value: float):
        self._normalize = value

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        file = str(self._data_paths[idx])
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(file)
        img = sitk.GetArrayFromImage(ifr.Execute()).astype('float32' if self._as_float32 else 'float64')
        return {"paths": str(file), "data": self.__normalize(img)}

    def __normalize(self, img: np.ndarray, maximum_1: bool = True) -> np.ndarray:
        if self._normalize > 0:
            norm = np.zeros(img.shape) + self._normalize
            if maximum_1:
                return np.clip(np.divide(img, norm), 0, 1)
            return np.divide(img, norm)
        return img


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

    def __iter__(self):
        return super().__iter__()


def preprocess_as_variable(image: np.ndarray, acceleration: float = 4.0) -> (
        torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor):
    im_und, k_und, mask, im_gnd = preprocess(image, acceleration)
    im_u = Variable(im_und.type(Module.TensorType))
    k_u = Variable(k_und.type(Module.TensorType))
    mask = Variable(mask.type(Module.TensorType))
    gnd = Variable(im_gnd.type(Module.TensorType))

    return im_u, k_u, mask, gnd


def preprocess(image: np.ndarray, acceleration: float = 4.0) -> (
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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
    im_gnd_l = to_tensor_format(image)
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und, complex=True))
    mask_l = torch.from_numpy(to_tensor_format(mask))

    return im_und_l, k_und_l, mask_l, im_gnd_l
