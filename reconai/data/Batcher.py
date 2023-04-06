import logging, re, os
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import rotate as scipy_rotate

from .Volume import Volume


class Sequence:
    def __init__(self, case: str, seq: Dict[str, List[int]]):
        self._sequence = seq
        for key in seq.keys():
            if not re.search('^\\d+_\\d+$', key):
                raise KeyError(f"{key} in {case} sequence is not of format '\d+_\d+'!")
        self._case = case

    @property
    def case(self) -> str:
        return self._case

    def __len__(self):
        return sum([len(s) for s in self._sequence.values()])

    def __eq__(self, other) -> bool:
        if isinstance(other, Sequence):
            return other._sequence == self._sequence
        elif isinstance(other, dict):
            return other == self._sequence
        else:
            return False

    def __repr__(self):
        return repr(self._sequence)

    def items(self) -> Iterable[Tuple[int, List[int]]]:
        for key in sorted(self._sequence.keys()):
            yield int(re.search('\\d+$', key).group()), self._sequence[key]


class SequenceCollection:
    def __init__(self, sequences: Dict[str, List[Sequence]]):
        # {case: a list of {i_mha: sequences as slice ids}}
        self._sequences = sequences

    def __len__(self):
        return len([seq for seqs in self._sequences.values() for seq in seqs])

    def __eq__(self, other: Dict[str, List[Sequence]]) -> bool:
        return other == self._sequences

    def __repr__(self):
        return repr(self._sequences)

    def items(self) -> Iterable[Sequence]:
        for case in self._sequences.keys():
            for seq in self._sequences[case]:
                yield seq


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

                    assert values_len(c_sequence) == seq_len, SystemError(f'FATAL: length of candidate ({len(c_sequence)}) != seq_len ({seq_len})')
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


def crop_or_expand(seq: np.ndarray, shape: Tuple[int, int]):
    z, y, x = seq.shape
    _y, _x = shape
    split_n = lambda a: [a // 2 + (1 if a < a % 2 else 0) for _ in range(2)]
    split_x = split_n(np.abs(x - _x))
    split_y = split_n(np.abs(y - _y))

    if x > _x:
        seq = seq[:, split_x[0]:-split_x[1], :]
    elif x < _x:
        seq = np.pad(seq, ([0, 0], [0, 0], x), mode='edge')

    if y > _y:
        seq = seq[:, :, split_y[0]:-split_y[1]]
    elif y < _y:
        seq = np.pad(seq, ([0, 0], y, [0, 0]), mode='edge')
    return seq


def normalize(seq: np.ndarray, norm: float):
    z, y, x = seq.shape
    return np.divide(seq, np.zeros((y, x)) + norm)


def flip_ud_lr(seq: np.ndarray, *flips: str):
    for f in flips:
        if f == 'ud':
            seq = np.flipud(seq)
        elif f == 'lr':
            seq = np.fliplr(seq)
    return seq


def rotate(seq: np.ndarray, *rotate_deg: float):
    for s, img in enumerate(seq):
        if rotate_deg[s] != 0:
            # order 0 prefers 'moving' the pixels, order 5 prefers interpolation
            seq[s, :, :] = scipy_rotate(img, rotate_deg[s], reshape=False, mode='nearest', order=1)
    return seq


class Batcher1:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._processed_sequences = []
        self._indexes = []
        self._crop_expand_to = None
        self._norm = None

    def __len__(self):
        return len(self._processed_sequences)

    def append_sequence(self, sequence: Sequence, *,
                        crop_expand_to: Tuple[int, int] = (256, 256),
                        norm: float = 1,
                        flip: str = '',
                        rotate_degs: None | List[float] = None):
        """
        Append sequence to this batcher. Apply different norm/flip/rotate_deg in a for loop to multiply the dataset.

        Parameters
        ----------
        sequence: Sequence
            Sequence, likely iterated from SequenceCollection.items()
        crop_expand_to: (int, int) = (256, 256)
            Crop or expand all images in the sequence to width x height size (must be same across appends)
        norm: float=1
            Normalize images by 'norm' (must be same across appends)
        flip: str
            Flip the sequence either 'lr' or 'ud' or 'lrud' or 'udlr'
        rotate_degs: List[float]
            Rotate the image by rotate_degs. Length must equal length of sequence. None, for no rotation.
        """
        # precalculate some values
        rotate_degs = [r % 360 for r in rotate_degs] if rotate_degs else [0] * len(sequence)
        flips: List[str] = [r.group() for r in re.finditer('(ud)|(lr)', flip)]
        norm = max(norm, np.nextafter(0, 1))

        assert len(rotate_degs) == len(sequence), ValueError(f'length of rotate_degs ({len(rotate_degs)}) != length of sequence ({len(sequence)})')
        assert not self._crop_expand_to or self._crop_expand_to == crop_expand_to, ValueError(f'crop_expand_to ({crop_expand_to}) != previous crop_expand_to ({self._crop_expand_to})')
        assert not self._norm or self._norm == norm, ValueError(f'norm ({norm}) != previous norm ({self._norm})')
        self._crop_expand_to, self._norm = crop_expand_to, norm

        images: List[np.ndarray] = self._dataloader[sequence.case]
        sequence_images = np.empty((0,) + crop_expand_to)
        for img_ids, slice_ids in sequence.items():
            img_slices = images[img_ids][slice_ids].copy()

            img_slices = crop_or_expand(img_slices, crop_expand_to)
            img_slices = flip_ud_lr(img_slices, *flips)
            img_slices = rotate(img_slices, *rotate_degs)
            img_slices = normalize(img_slices, norm)

            sequence_images = np.vstack((sequence_images, img_slices))

        self._indexes.append(len(self._processed_sequences))
        self._processed_sequences.append(sequence_images)

    def shuffle(self, seed: int = -1):
        self.sort()
        r = np.random.default_rng(seed if seed >= 0 else None)
        r.shuffle(self._indexes)

    def sort(self):
        self._indexes.sort()

    def items(self) -> np.ndarray:
        """
        Retrieves np.ndarray sequences. If it is not shuffled, it will be in the same order as sequences were appended.
        """
        for i in self._indexes:
            yield self._processed_sequences[i]

    def items_fold(self, fold: int, max_folds: int = 1, validation: bool = False) -> np.ndarray:
        """
        Retrieves np.ndarray sequences. If it is not shuffled, it will be in the same order as sequences were appended.

        Parameters
        ----------
        fold: int
            fold < max_folds
        max_folds: int
            Keep this value constant to keep folds consistent
        validation: bool=False
            Return the validation set or the training set
        """
        assert max_folds >= 0, ValueError('max_folds must be at least 0')
        assert fold < max_folds, IndexError(f'fold ({fold}) >= max folds ({max_folds})')

        validation_ids = set(self._indexes[fold::max_folds])
        training_ids = validation_ids.symmetric_difference(self._indexes)

        for i in validation_ids if validation else training_ids:
            yield self._processed_sequences[i]


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
