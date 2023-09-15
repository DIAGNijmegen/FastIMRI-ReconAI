from pathlib import Path
from typing import List

import SimpleITK as sitk
import torch.utils.data as torch_data
import numpy as np


class Dataset(torch_data.Dataset):
    def __init__(self, data_dir: Path, *, normalize: float = 0, as_float32: bool = True):
        self._data_paths: List[Path] = [item.resolve() for item in data_dir.iterdir() if item.suffix == '.mha']
        self._data_len = len(self._data_paths)
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
        file = self._data_paths[idx]
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(file))
        img = sitk.GetArrayFromImage(ifr.Execute()).astype('float32' if self._as_float32 else 'float64')
        return self.__normalize(img)

    def __normalize(self, img: np.ndarray, maximum_1: bool = True) -> np.ndarray:
        if self._normalize > 0:
            norm = np.zeros(img.shape) + self._normalize
            if maximum_1:
                return np.clip(np.divide(img, norm), 0, 1)
            return np.divide(img, norm)
        return img


class DataLoader(torch_data.DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
