import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import SimpleITK as sitk
import numpy as np


class DataLoader:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        # {case: a list of mhas}
        self._mhas: Dict[str, List[np.ndarray]] = {}

    def __len__(self):
        return sum(len(v) for v in self._mhas.values())

    def __getitem__(self, item: str) -> List[np.ndarray]:
        return [x.copy() for x in self._mhas[item]]

    def keys(self):
        return self._mhas.keys()

    def shapes(self, item: str) -> List[tuple]:
        return [m.shape for m in self._mhas[item]]

    def load(self, split_regex: str, *, filter_regex: str = '', as_float32: bool = True):
        """
        Parameters
        ----------
        split_regex: str
            first capture group is how to split mhas into cases (eg. '.*_(.*)_')
        filter_regex: str=''
            result is loaded
        as_float32: bool=True
            data is either float32 (True) or float64 (False)
        """
        all_dirs = set()
        for root, _, files in os.walk(self._data_dir):
            all_dirs.add(Path(root))
        all_dirs.remove(self._data_dir)

        def gather_mhas(dirname: Path):
            mhas = {}
            for file in dirname.iterdir():
                search = re.search(filter_regex, file.name) if filter_regex else True
                split = re.search(split_regex, file.name)
                if search and split and split.groups() and file.name.endswith('.mha'):
                    mha = self._load_mha(dirname / file.name).astype('float32' if as_float32 else 'float64')
                    key = (dirname / split.group(1)).as_posix()
                    mhas[key] = mhas.get(key, []) + [mha]
            return mhas

        # with ThreadPoolExecutor(max_workers=4) as pool:
        #     futures = {pool.submit(gather_mhas, d): d for d in all_dirs}
        #     for future in as_completed(futures):
        #         self._mhas.update(future.result())
        for d in all_dirs:
            self._mhas.update(gather_mhas(d))

    @staticmethod
    def _load_mha(mha: Path) -> np.ndarray:
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(mha.resolve()))
        return sitk.GetArrayFromImage(ifr.Execute())
