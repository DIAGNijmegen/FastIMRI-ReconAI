import logging
from pathlib import Path
from typing import List

import SimpleITK as sitk
import numpy as np


class Volume:
    key: str = 'sag'
    shape: int = 256  # intended shape of image (w x h)
    sequence_length: int = 15  # length of total sequence
    slicing: int = 3  # slices to take from image and add to sequence
    sequence_shift: float = 2/3

    def __init__(self, study_id: str, files: List[Path]):
        if not (self.sequence_length / self.slicing).is_integer():
            raise ValueError(f'{self.sequence_length}÷{self.slicing} is not an integer')

        self.study_id = study_id
        self.files = list(sorted([file for file in files if self.key in file.name]))

        self._is_valid = len(self.files) * self.slicing < self.sequence_length
        if not self._is_valid:
            logging.warning(f'insufficient data in case {self.study_id} to reach a {self.sequence_length} sequence. '
                            f'({len(self.files)} * {self.slicing} < {self.sequence_length}')

    def __repr__(self):
        return f'{self.study_id} ({len(self.files)} files)'

    @staticmethod
    def load(file: Path) -> np.ndarray:
        ifr = sitk.ImageFileReader()
        ifr.SetFileName(str(file))
        return sitk.GetArrayFromImage(ifr.Execute())

    def to_ndarray(self, norm: float = 1, slicing: List[str] = None) -> np.ndarray:
        images = dict()

        volumes, t, rev = [], 0, False
        while t + self.sequence_length <= len(self.files) * self.slicing:
            sequence = []
            i = 0
            for file in self.files[t // self.slicing:(t + self.sequence_length) // self.slicing]:
                images[file] = images.get(file, self.load(file).astype('float64'))
                img = images[file]
                # do we guarantee to take the center slice? do we take all 5 slices, or less? just one? what order?
                z = img.shape[0]
                if z < self.slicing:
                    raise ValueError(f'{z} < {self.slicing}, cannot split image up ({file}')
                img = ensure_correct_image_shape(img, self.shape)

                # randoms, center, randoms
                slices = self._interpret_slicing(z, slicing)
                sequence.extend([img[s, :, :] / norm for s in slices])
                # slices = np.random.choice(list(set(range(z)).difference([z // 2])), size=self.slicing, replace=False)
                # slices[len(slices) // 2] = z // 2
                # for s in sorted(slices):
                #     # s_im = img[s, :, :] / 1961.06
                #     # s_im[(64 * i):(64 * (i+1)), (64 * i):(64 * (i+1))] = 0
                #     # sequence.append(s_im)
                #     # i += 1
                #     sequence.append(img[s, :, :] / norm)

            # sequence = list(reversed(sequence)) if rev else sequence
            volumes.append(sequence)

            rev = not rev
            t += int(self.sequence_shift * self.sequence_length)

        # CODE to switch slideshow to 1
        # Uncomment while-loop above
        # self._ifr.SetFileName(str(files[0]))
        # img = sitk.GetArrayFromImage(self._ifr.Execute()).astype('float64')
        #
        # # Get middle image
        # z = img.shape[0]
        # sequence = []
        # for i in range(self.sequence_length):
        #     sequence.append(img[z//2, :, :] / 1961.06)
        #
        # volumes.append(sequence)

        # sanity check
        if not all(len(s) == self.sequence_length for s in volumes):
            raise ValueError(f'not all sequences are equal to {self.sequence_length}')
        return np.stack(volumes)

    def _interpret_slicing(self, z, slicing: List[str]):
        if len(slicing) != self.slicing:
            raise ValueError(f'slicing specifier {slicing} is not of length {self.slicing}')
        #['r', 'c', 'r']



def split_n(a) -> List[int]:
    return [a // 2 + (1 if a < a % 2 else 0) for _ in range(2)]


def ensure_correct_image_shape(image: np.ndarray, shape: int):
    z, y, x = image.shape
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
