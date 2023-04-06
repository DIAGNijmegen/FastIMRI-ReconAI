import re
from typing import List, Tuple

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from .sequence import Sequence
from .dataloader import DataLoader


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

        assert len(rotate_degs) == len(sequence), \
            ValueError(f'length of rotate_degs ({len(rotate_degs)}) != length of sequence ({len(sequence)})')
        assert not self._crop_expand_to or self._crop_expand_to == crop_expand_to, \
            ValueError(f'crop_expand_to ({crop_expand_to}) != previous crop_expand_to ({self._crop_expand_to})')
        assert not self._norm or self._norm == norm, \
            ValueError(f'norm ({norm}) != previous norm ({self._norm})')
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
