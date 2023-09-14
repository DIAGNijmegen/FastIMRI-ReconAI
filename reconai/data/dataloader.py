import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Callable

import SimpleITK as sitk
import numpy as np

from .sequence import Sequence, SequenceCollection


def _len_of_values(obj: dict):
    return sum([len(x) for x in obj.values()])


class DataLoader:
    MAX_WORKERS = 8

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

    def load(self, split_regex: str, *, filter_regex: str = '', as_float32: bool = True) -> 'DataLoader':
        """
        Load all data into memory.

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
        ifr = sitk.ImageFileReader()
        for root, _, files in os.walk(self._data_dir):
            all_dirs.add(Path(root))

        def gather_mhas(path: Path):
            mhas = dict()
            for file in path.iterdir():
                search = re.search(filter_regex, file.name) if filter_regex else True
                split = re.search(split_regex, file.name)
                if search and split and split.groups() and file.name.endswith('.mha'):
                    path_mha = path / file.name
                    ifr.SetFileName(str(path_mha.resolve()))
                    mha = sitk.GetArrayFromImage(ifr.Execute()).astype('float32' if as_float32 else 'float64')
                    key = (path / split.group(1)).as_posix()
                    mhas[key] = mhas.get(key, []) + [mha]
            return mhas

        # with ThreadPoolExecutor() as pool:
        #     futures = {pool.submit(gather_mhas, d): d for d in all_dirs}
        #     for future in as_completed(futures):
        #         self._mhas.update(future.result())
        for dire in all_dirs:
            self._mhas.update(gather_mhas(dire))

        return self

    def _generate_sequences(self, generate_func: Callable[[int, str], List[Sequence]]) -> SequenceCollection:
        sequence_collection: Dict[str, List[Sequence]] = {}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {pool.submit(generate_func, key): key for key in self.keys()}
            for future in as_completed(futures):
                key = futures[future]
                sequence_collection[key] = future.result()

        return SequenceCollection(sequence_collection)

    def generate_sequences(self, *, seed: int = -1, seq_len: int = 5) \
            -> SequenceCollection:
        """
        For each MHA slice, a sequence is created. If seq_len < len(MHA slice), the center slices are preferred.
        If seq_len > len(MHA slice), we perform a wrap around. The result is always equal to the number of MHA files.
        Parameters
        ----------
        seed:
            Seed to use for entire sequence generation
        seq_len: int=5
            Length of sequence
        """
        seeds = {case: seed + s for s, case in enumerate(self.keys())}

        def _generate_sequences(case: str) -> List[Sequence]:
            r = np.random.default_rng(seeds[case] if seed >= 0 else None)
            sequences: List[Sequence] = []
            shapes = self.shapes(case)
            slices = {m: np.array(list(range(shapes[m][0]))) for m in range(len(shapes))}

            for i, sl in slices.items():
                sl_len = len(sl)
                if seq_len > sl_len:
                    seq_left = (sl_len - seq_len) // 2
                    seq_right = sl_len - seq_len - seq_left
                    sl = np.pad(sl, (-seq_left, -seq_right), mode='reflect')
                    if seq_len != len(sl):
                        logging.info(f'case with seq_len > len(sl) wrong pad. {case}')
                        break
                elif seq_len < sl_len:
                    seq_left = (sl_len - seq_len) // 2
                    seq_right = sl_len - seq_len - seq_left
                    sl = sl[seq_left:seq_len + seq_right]

                if seq_len != len(sl):
                    logging.error(f'case with seq_len != len(sl). {case}')
                    break
                sequences.append(Sequence(case, {f'0_{i}': sl}))

            return sequences

        return self._generate_sequences(_generate_sequences)