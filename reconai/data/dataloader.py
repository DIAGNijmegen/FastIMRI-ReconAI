import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import SimpleITK as sitk
import numpy as np

from .sequence import Sequence, SequenceCollection


class DataLoader:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        # {case: a list of mhas}
        self._mhas: Dict[str, List[np.ndarray]] = {}

    def __len__(self):
        return sum(len(v) for v in self._mhas.values())

    def __getitem__(self, item: str) -> List[np.ndarray]:
        return [x.copy() for x in self._mhas[item]]

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

        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(gather_mhas, d): d for d in all_dirs}
            for future in as_completed(futures):
                self._mhas.update(future.result())

    @staticmethod
    def _load_mha(mha: Path):
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(mha.resolve()))
        return sitk.GetArrayFromImage(ifr.Execute())

    def generate_sequences_from_dataset(self, *, seed: int = -1, seq_len: int = 15, mean_slices_per_mha: float = 2,
                                        max_slices_per_mha: int = 3, q: float = 0.5) -> SequenceCollection:
        """
        Every MHA slice within a case has a 'usefulness' score, calculated as 1/2^x, where is the number of steps away
        from the center slice. For each MHA slice, a sequence takes 1 to (max_slices_per_mha)
        Parameters
        ----------
        seed:
            Seed to use for entire sequence generation
        seq_len: int=15
            Length of sequence
        mean_slices_per_mha: float
            Mean number of slices to take for each .mha
        max_slices_per_mha: int
            Max number of slices to take for each .mha
        q: float=1.75/3
            Quality preference. (0 <= q <= 1)
            Higher means fewer sequences per case, with each sequence containing primarily center slices.
            Lower means more sequences per case, with each sequence containing primarily edge slices.
        """
        q = np.clip(q, 0, 1) * seq_len
        values_len = lambda a: sum([len(x) for x in a.values()])
        seeds = {case: seed + s for s, case in enumerate(self._mhas.keys())}

        def _generate_sequences(case: str, images: List[np.ndarray]) -> List[Sequence]:
            r = np.random.default_rng(seeds[case] if seed >= 0 else None)
            sequences: List[Sequence] = []
            range_len_images = range(len(images))
            available = {m: list(range(images[m].shape[0])) for m in range_len_images}
            slices = {m: np.array(available[m]) for m in range_len_images}

            while True:
                sequence_candidates: list = []

                # is it impossible to make a full sequence, given the number of available slices?
                if values_len(available) < seq_len:
                    break

                # generate 100 sequence candidates
                for _ in range(100):
                    c_quality = 0
                    c_sequence = {}
                    c_available = deepcopy(available)

                    # while our candidate sequence is not a full sequence of length seq_len
                    m, loop = 0, 0
                    while (c_sequence_len := values_len(c_sequence)) < seq_len:
                        m_available_slices = len(c_available[m])
                        if m_available_slices > 0:
                            # we take n_slices, normally distributed around mean_slices_per_mha
                            max_n_slices = min(m_available_slices, max_slices_per_mha, seq_len - c_sequence_len)
                            n_slices = int(np.clip(np.round(r.normal(loc=mean_slices_per_mha)), 0, max_n_slices))

                            # quality of a slice is 1/2^x, where x is x steps away from center
                            m_quality = 1 / np.power(2, np.abs(slices[m] - 2))

                            # probability distribution are 1 unless..
                            m_p = np.ones(slices[m].shape)
                            if n_slices == 1:
                                # greatly increase chance of taking a center slice if only taking one
                                m_p = m_quality
                            m_p *= [(x in c_available[m]) for x in slices[m]]

                            selected_slices = r.choice(slices[m], size=int(n_slices), replace=False, p=m_p / m_p.sum())

                            c_quality += m_quality[selected_slices].sum()
                            id = f'{loop}_{m}'
                            c_sequence[id] = c_sequence.get(id, []) + selected_slices.tolist()
                            [c_available[m].remove(s) for s in selected_slices]

                        m = (m + 1) % len(c_available)
                        if m == 0:
                            loop += 1

                    assert values_len(c_sequence) == seq_len, \
                        SystemError(f'FATAL: length of candidate ({len(c_sequence)}) != seq_len ({seq_len})')
                    sequence_candidates.append((c_quality, Sequence(case, c_sequence), c_available))

                # select candidate closest to quality preference 'q'
                sequence_candidates.sort(key=lambda c: np.abs(q - c[0]))
                c_quality, c_sequence, c_available = sequence_candidates[0]
                if c_quality < q:
                    break
                else:
                    sequences.append(c_sequence)
                    available = c_available
            return sequences

        sequence_collection: Dict[str, List[Sequence]] = {}
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(_generate_sequences, key, images): key for key, images in self._mhas.items()}
            for future in as_completed(futures):
                key = futures[future]
                sequence_collection[key] = future.result()

        return SequenceCollection(sequence_collection)
