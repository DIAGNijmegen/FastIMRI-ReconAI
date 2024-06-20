from pathlib import Path
from typing import List

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import monai

import reconai.math.compressed_sensing as cs
from reconai.math.kspace import get_rand_exp_decay_mask
from reconai.model.dnn_io import to_tensor_format
from reconai.model.module import Module
from reconai.parameters import Parameters


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, params: Parameters, as_float32: bool = True):
        self._data_paths: List[Path] = [item.resolve() for item in data_dir.iterdir() if
                                        item.suffix in ['.npy', '.mha']]

        self._as_float32 = as_float32
        self._params = params
        if len(self._data_paths) == 0:
            raise ValueError(f'no .mha files found in {data_dir}!')

        img, _, _, _ = self._image(self._data_paths[0])
        z, y, x = img.shape
        seq = params.data.sequence_length
        self._step1 = monai.transforms.Compose([
            monai.transforms.SpatialPad([z, max(y, params.data.shape_y), max(x, params.data.shape_x)]),
            monai.transforms.ScaleIntensity(),
            monai.transforms.RandRicianNoise(1, 0.005, 0.005)
        ], lazy=True)
        self._step2 = monai.transforms.SomeOf([
            monai.transforms.RandFlip(prob=1, spatial_axis=-1),
            monai.transforms.RandFlip(prob=1, spatial_axis=-2),
            monai.transforms.OneOf([monai.transforms.Rotate(np.pi * r / 2) for r in range(-3, 4) if r != 0])
        ], lazy=True, weights=[50, 50, 75], num_transforms=(0 if self._params.train.augment_all else 1, 3))

        if seq > 1:
            self._data_len = len(self._data_paths)
            self._s = (z - seq) // 2
            self._e = self._s + seq
        else:
            self._data_paths = self._data_paths * z
            self._data_len = len(self._data_paths)
            self._s, self._e = z, z

    def __len__(self):
        return self._data_len * max(1, self._params.train.augment_mult)

    def __getitem__(self, idx):
        index = idx // self._params.train.augment_mult if idx >= self._data_len else idx
        try:
            file = str(self._data_paths[index])
        except:
            raise ValueError(f'oh no: {index} {idx}')
        img, origin, direction, spacing = self._image(file)
        item = {"paths": file, "origin": origin, "direction": direction, "spacing": spacing}

        img = self._step1(torch.from_numpy(np.expand_dims(img, axis=0)))
        if idx >= self._data_len or self._params.train.augment_all:
            img = self._step2(img)

        if self._s == self._e:
            i = int(idx // (len(self) / self._s))
            return item | {"data": img.numpy().squeeze(axis=0)[i:i+1], "slice": i}
        else:
            return item | {"data": img.numpy().squeeze(axis=0)[self._s:self._e], "slice": -1}

    def _image(self, file: Path | str) -> tuple[np.ndarray, tuple, tuple, tuple]:
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(file))
        return (sitk.GetArrayFromImage(ifr.Execute()).astype('float32' if self._as_float32 else 'float64'),
                ifr.GetOrigin(), ifr.GetDirection(), ifr.GetSpacing())


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, indices: list[int] | int = 0):
        if isinstance(indices, int):
            sampler = torch.utils.data.RandomSampler(dataset, num_samples=indices if indices > 0 else None)
        else:
            sampler = torch.utils.data.SubsetRandomSampler(list(indices))
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    def __iter__(self):
        return super().__iter__()


def preprocess_real(image: np.ndarray, k: np.ndarray, mask: np.ndarray) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    assert len(image.shape) == 4

    if mask.shape == image.shape:
        pass
    elif mask.ndim == 1 and mask.shape[0] == image.shape[2]:
        mask = np.tile(mask[None, None, :, None], (*image.shape[:2], 1, image.shape[3]))
    elif mask.ndim == 2 and mask.shape == image.shape[2:]:
        # untested
        mask = np.tile(mask[None, None, :, :], (*image.shape[:2], 1, 1))
    elif mask.ndim == 3 and mask.shape == image.shape[3:]:
        mask = np.expand_dims(mask, axis=0)
    else:
        mask = np.ones(shape=image.shape, dtype=np.uint8)

    im_und_l = Variable(torch.from_numpy(to_tensor_format(image)).type(Module.TensorType))
    k_und_l = Variable(torch.from_numpy(to_tensor_format(k, complex=True)).type(Module.TensorType))
    mask_l = Variable(torch.from_numpy(to_tensor_format(mask)).type(Module.TensorType))

    return im_und_l, k_und_l, mask_l


def preprocess_simulated(image: np.ndarray, acceleration: float = 4.0) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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
    mask = np.ones(image.shape)

    if acceleration > 1:
        for b_ in range(b):
            for s_ in range(s):
                mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / acceleration, 1 / 3)
    im_und, k_und = cs.undersample(image, mask, centred=True, norm='ortho')
    from reconai.math.fourier import fft2c, ifft2c
    for k in range(k_und.shape[1]):
        k_und[0, k] = fft2c(np.abs(ifft2c(k_und[0, k])))
    im_gnd_l = Variable(torch.from_numpy(to_tensor_format(image)).type(Module.TensorType))
    im_und_l = Variable(torch.from_numpy(to_tensor_format(im_und)).type(Module.TensorType))
    k_und_l = Variable(torch.from_numpy(to_tensor_format(k_und, complex=True)).type(Module.TensorType))
    mask_l = Variable(torch.from_numpy(to_tensor_format(mask)).type(Module.TensorType))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def image(array: np.ndarray) -> np.ndarray:
    array_min, array_max = array.min(), array.max() + np.finfo(array.dtype).eps
    array_norm = (array - array_min) / (array_max - array_min)
    return (np.clip(array_norm, 0, 1) * 255).astype(np.uint8)
