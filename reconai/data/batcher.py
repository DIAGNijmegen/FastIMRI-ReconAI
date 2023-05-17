import random
import re
from typing import List, Tuple

import numpy as np
from scipy.ndimage import rotate as scipy_rotate, zoom as scipy_zoom

from .sequence import Sequence
from .dataloader import DataLoader


class Batcher:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader

        self._processed_sequences = []
        self._indexes = []

        self._crop_expand_to = None
        self._norm = None

    def __len__(self):
        return len(self._processed_sequences)

    def append_sequence(self, sequence: Sequence,
                        crop_expand_to: Tuple[int, int] = (256, 256),
                        norm: float = 1,
                        flip: str = '',
                        zoom_factor: float = 1,
                        rotate_deg: float = 0,
                        equal_images: bool = False,
                        expand_to_n: bool = False):
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
        zoom_factor: float
            Zoom the images by 'zoom_factor'.
        rotate_deg: float
            Rotate the images by 'rotate_deg's'.
        equal_images: bool
            If set to true, then repeat first image all the time
        expand_to_n: bool
            If true and equal_images is also true, then it will expand the sequence n times.
        """
        # precalculate some values
        flips: List[str] = [r.group() for r in re.finditer('(ud)|(lr)', flip)]
        norm = max(norm, np.nextafter(0, 1))

        assert not self._crop_expand_to or self._crop_expand_to == crop_expand_to, \
            ValueError(f'crop_expand_to ({crop_expand_to}) != previous crop_expand_to ({self._crop_expand_to})')
        assert not self._norm or self._norm == norm, \
            ValueError(f'norm ({norm}) != previous norm ({self._norm})')
        self._crop_expand_to, self._norm = crop_expand_to, norm

        data: List[np.ndarray] = self._dataloader[sequence.case]
        data = [data[img_ids][slice_ids] for img_ids, slice_ids in sequence.items()]

        sequence_images = np.empty((0,) + crop_expand_to)
        for slices in data:
            for img in slices:
                img = rotate(img, rotate_deg)
                img = zoom(img, zoom_factor)
                img = crop_or_pad(img, crop_expand_to)
                img = flip_ud_lr(img, *flips)
                img = normalize(img, norm)
                sequence_images = np.vstack((sequence_images, img[np.newaxis, ...]))

        if equal_images and expand_to_n:
            new_sequence = sequence_images.copy()
            for i in range(0, len(sequence_images)):
                for j in range(0, len(sequence_images)):
                    new_sequence[j] = sequence_images[i]
                self.add_processed_sequence(new_sequence)
        else:
            if equal_images:
                randint = random.randint(0, len(sequence_images) - 1)
                for i in range(0, len(sequence_images)):
                    sequence_images[i] = sequence_images[randint]
            self.add_processed_sequence(sequence_images)

    def add_processed_sequence(self, processed_sequence: np.ndarray):
        self._indexes.append(len(self._processed_sequences))
        self._processed_sequences.append(processed_sequence)

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
            yield np.stack([self._processed_sequences[i]])

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
        assert max_folds > 2, ValueError('max_folds must be at least 2, use items() instead')
        assert fold < max_folds, IndexError(f'fold ({fold}) >= max folds ({max_folds})')

        validation_ids = set(self._indexes[fold::max_folds])
        training_ids = validation_ids.symmetric_difference(self._indexes)

        for i in validation_ids if validation else training_ids:
            yield np.stack([self._processed_sequences[i]])


def crop_or_pad(img: np.ndarray, target_shape: Tuple[int, int]):
    y, x = img.shape
    _y, _x = target_shape
    split_n = lambda a: [a // 2 + (1 if a < a % 2 else 0) for _ in range(2)]
    split_x = split_n(np.abs(x - _x))
    split_y = split_n(np.abs(y - _y))

    if x > _x:
        img = img[:, split_x[0]:-split_x[1]]
    elif x < _x:
        img = np.pad(img, [(0, 0), split_x], 'edge')

    if y > _y:
        img = img[split_y[0]:-split_y[1], :]
    elif y < _y:
        img = np.pad(img, [split_y, (0, 0)], 'edge')
    return img


def normalize(img: np.ndarray, norm: float, maximum1: bool = True):
    if maximum1:
        return np.clip(np.divide(img, np.zeros(img.shape) + norm), 0, 1)
    return np.divide(img, np.zeros(img.shape) + norm)


def flip_ud_lr(img: np.ndarray, *flips: str):
    for f in flips:
        if f == 'ud':
            img = np.flipud(img)
        elif f == 'lr':
            img = np.fliplr(img)
    return img


def zoom(img: np.ndarray, zoom: float):
    return scipy_zoom(img, zoom, order=1)


def rotate(img: np.ndarray, rotate_deg: float):
    return scipy_rotate(img, rotate_deg % 360, reshape=False, mode='nearest', order=1)

