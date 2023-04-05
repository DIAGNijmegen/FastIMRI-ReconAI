import logging, re, os
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

from .Volume import Volume

Sequences = List[Dict[str, List[int]]]


class SequenceCollection:
    def __init__(self, sequences: Dict[str, Sequences]):
        # {case: a list of {mha: sequences as slice ids}}
        self._sequences = sequences

    def __eq__(self, other: Dict[str, Sequences]) -> bool:
        return other == self._sequences

    def __getitem__(self, item: str) -> Sequences:
        return deepcopy(self._sequences[item])

    def keys(self):
        return self._sequences.keys()


class DataLoader:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        # {case: a list of mhas}
        self._mhas: Dict[str, List[np.ndarray]] = {}

    def __getitem__(self, item: str) -> List[np.ndarray]:
        return [x.copy() for x in self._mhas[item]]

    def load(self, split_regex: str, *, filter_regex: str = '', as_float32: bool = True):
        """
        Parameters
        ----------
        split_regex: str
            first capture group is how to split mhas into cases (eg. '.*_(.*)_')
        filter_regex: str=''
            result is loaded in the Batcher
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
                if search and split.groups() and file.name.endswith('.mha'):
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

    def generate_sequences(self, *, seed: int = -1, seq_len: int = 15, mean_slices_per_mha: float = 2,
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

        def _generate_sequences(s_seed: int, images: List[np.ndarray]) -> Sequences:
            r = np.random.default_rng(s_seed if seed >= 0 else None)
            sequences = []
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

                    # seq[0], [1], [2]
                    sequence_candidates.append((c_quality, c_sequence, c_available))

                # select candidate closest to quality preference 'q'
                sequence_candidates.sort(key=lambda c: np.abs(q - c[0]))
                seq = sequence_candidates[0]
                if seq[0] < q:
                    break
                else:
                    sequences.append(seq[1])
                    available = seq[2]
            return sequences

        sequences: Dict[str, Sequences] = {}
        with ThreadPoolExecutor() as pool:
            seeds = {key: seed + s for s, key in enumerate(self._mhas.keys())}
            futures = {pool.submit(_generate_sequences, seeds[key], images): key for key, images in self._mhas.items()}
            for future in as_completed(futures):
                key = futures[future]
                sequences[key] = future.result()
        return SequenceCollection(sequences)


def crop_or_expand_to(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    z, y, x = image.shape
    split_n = lambda a: [a // 2 + (1 if a < a % 2 else 0) for _ in range(2)]
    split_x = split_n(np.abs(x - shape))
    split_y = split_n(np.abs(y - shape))

    if x > shape:
        image = image[:, split_x[0]:-split_x[1], :]
    elif x < shape:
        image = np.pad(image, ([0, 0], [0, 0], x), mode='edge')

    if y > shape:
        image = image[:, :, split_y[0]:-split_y[1]]
    elif y < shape:
        image = np.pad(image, ([0, 0], y, [0, 0]), mode='edge')

    return image


class Batcher1:
    def __init__(self, dataloader: DataLoader, n_folds: int = 1):
        self._dataloader = dataloader
        self._n_folds = n_folds
        self._numpy_sequences = None

    def append_sequences(self, sequence_collection: SequenceCollection, *, norm: float = 1, flip: str = '',
                         rotate_deg: int = 0, crop_expand_to: Tuple[int, int] = (256, 256)):
        rotate_deg = rotate_deg % 360
        for case in sequence_collection.keys():
            images = self._dataloader[case]
            sequences = sequence_collection[case]
            for sequence in sequences:
                # loop through 0_0 .. 0_n until doesn't exist, then 1_0 .. 1_n until k_0 .. k_n
                pass

        pass

class Batcher:
    batch_size: int = 1
    shuffle: bool = True

    def __init__(self, volumes: List[Volume]):
        self.volumes: List[Volume | np.ndarray] = list(volumes)
        self._blacklist = []

    def get_blacklist(self):
        for _ in self.generate():
            pass
        return self._blacklist

    def generate(self) -> np.ndarray:
        minibatch = []
        all_array = False  # self.volumes only has array elements
        while not (all_array and len(self.volumes) < self.batch_size - len(minibatch)):
            nv = len(self.volumes)

            # convert all self.volumes to arrays when it is small, while loop conditional takes effect
            if not all_array and nv <= self.batch_size:
                for item in self.volumes:
                    if isinstance(item, Volume):
                        self.volumes.remove(item)
                        try:
                            data = item.to_ndarray()
                        except Exception as e:
                            logging.warning(str(e))
                            self._blacklist.append(item.study_id)
                            continue
                        else:
                            self.volumes.extend([data[d] for d in reversed(range(len(data)))])
                all_array = True

            # select a random from volume
            nex = np.random.choice(range(nv)) if Batcher.shuffle else nv - 1
            try:
                item = self.volumes.pop(nex)
            except IndexError as e:
                logging.warning(str(e))
                break

            if isinstance(item, Volume):
                # select random array if volume splits into multiple, rest is put back
                try:
                    data = item.to_ndarray()
                except Exception as e:
                    logging.warning(str(e))
                    self._blacklist.append(item.study_id)
                    continue
                else:
                    nex = np.random.choice(range(len(data))) if Batcher.shuffle else 0
                    for d in reversed(range(len(data))):
                        if d == nex:
                            minibatch.append(data[d])
                        else:
                            self.volumes.append(data[d])
            else:
                data = item
                minibatch.append(data)

            if len(minibatch) == Batcher.batch_size:
                yield np.stack(minibatch)
                minibatch = []
