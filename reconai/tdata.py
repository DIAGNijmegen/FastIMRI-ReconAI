from pathlib import Path
from typing import List

import SimpleITK as sitk
import torch.utils.data as torch_data


class Dataset(torch_data.Dataset):
    def __init__(self, data_dir: Path, *, as_float32: bool = True):
        self._data_paths: List[Path] = [item.resolve() for item in data_dir.iterdir() if item.suffix == '.mha']
        self._data_len = len(self._data_paths)
        self._as_float32 = as_float32

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        file = self._data_paths[idx]
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(file))
        return sitk.GetArrayFromImage(ifr.Execute()).astype('float32' if self._as_float32 else 'float64')


class DataLoader(torch_data.DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
