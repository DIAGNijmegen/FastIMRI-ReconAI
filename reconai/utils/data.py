import numpy as np
from typing import List, Tuple
from os.path import join

import torch
import SimpleITK as sitk
from pathlib import Path

from .kspace import get_rand_exp_decay_mask
import reconai.utils.compressed_sensing as cs
from reconai.cascadenet_pytorch.dnn_io import to_tensor_format, from_tensor_format
import matplotlib.pyplot as plt


def prepare_input(image: np.ndarray, acceleration: float = 4.0) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    image: ndarray - input image of shape (batch_size, n_channels, width, height)
    acceleration: float - controls the undersampling rate. higher the value, more undersampling

    Returns
    ------
    im_und_l: Tensor - undersampled image in image space
    k_und_l: Tensor - undersampled image in K-space
    mask_l: Tensor - undersampling mask in fourier domain (which lines in k-space to keep / which to ignore)
    im_gnd_l: Tensor - ground truth image in image space
    """
    b, s, y, x = image.shape
    mask = np.zeros(image.shape)
    for b_ in range(b):
        for s_ in range(s):
            mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / acceleration, 1 / 3)

    im_und, k_und = cs.undersample(image, mask, centred=True, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(image))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data: np.ndarray, batch_size: int, shuffle: bool = True):
    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


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


def prepare_case(case: Tuple[str, List[Path]],
                 key: str = "sag",
                 shape: int = 256,
                 sequence_length: int = 15,
                 slice_count: int = 3,
                 sequence_shift: float = 2/3
                 ) -> np.ndarray:
    study_id, files = case
    originalfiles = files
    files = list(sorted([file for file in files if key in file.name]))  # filter the files where 'Needle' not in name
    if len(files) * slice_count < sequence_length:
        raise ValueError(f'insufficient data in case {study_id} to reach a sequence of length {sequence_length}. \n {originalfiles[0].name}')

    ifr = sitk.ImageFileReader()
    images, volumes, t, rev = dict(), [], 0, False
    while t + sequence_length <= len(files) * slice_count:
        sequence = []
        for file in files[t // slice_count: (t+sequence_length) // slice_count]:
            ifr.SetFileName(str(file))
            images[file] = images.get(file, sitk.GetArrayFromImage(ifr.Execute()).astype('float64'))
            img = images[file]

            z = img.shape[0]
            if z <= slice_count:
                print(f'{file} has too few images to create slice')
                continue
                # raise ValueError(f'{z} < {slice_count}, cannot split image up')

            img = ensure_correct_image_shape(img, shape)

            # randoms, center, randoms
            slices = np.random.choice(list(set(range(z)).difference([z // 2])), size=slice_count, replace=False)
            slices[len(slices) // 2] = z // 2
            for s in sorted(slices):  # 3 2 0 example. Do we want to have the slices in ascending order?
                sequence.append(img[s, :, :])

        sequence = list(reversed(sequence)) if rev else sequence
        volumes.append(sequence)

        rev = not rev
        t += int(sequence_shift * sequence_length)

    # sanity check
    if not all(len(s) == sequence_length for s in volumes):
        raise ValueError(f'not all sequences are equal to {sequence_length}')
    return np.stack(volumes)


def generate_volumes(data_dir: Path):
    for patient_dir in data_dir.iterdir():
        try:
            if patient_dir.is_dir():
                files = list(patient_dir.iterdir())
                study_ids = {fn.name.split('_')[1] for fn in files}
                for study_id in study_ids:
                    yield study_id, [fn for fn in files if study_id in fn.name]
        except:
            continue

def create_or_load_datasets(sequence_length: int,
                            data_dir: Path,
                            filename: str = 'dataset.npy',
                            force_create: bool = False
                            ) -> np.ndarray:
    file = join(data_dir, filename)
    if not force_create:
        try:
            data = np.load(file)
            return data
        except:
            print('load failed')

    shape = 256
    volumes = generate_volumes(data_dir)
    cases = []
    for volume in volumes:
        try:
            cases.append(prepare_case(volume, shape=shape, sequence_length=sequence_length))
        except:
            continue
    data = np.concatenate(cases)
    np.save(file, data)
    return data


def generate_datasets(batch_size: int, sequence_length: int, data_dir: Path) -> (np.ndarray, np.ndarray, np.ndarray):
    data = create_or_load_datasets(sequence_length, data_dir, force_create=False)
    print('succesfully loaded data')
    # Split data
    data_split = np.array_split(range(len(data)), 10)
    if len(data_split) < 3 and len(data) // 3 >= batch_size:
        raise ValueError('insufficient data')

    train = np.asarray([data[i] for i in np.concatenate(data_split[2:])])
    test = np.asarray([data[i] for i in data_split[0]])
    validate = np.asarray([data[i] for i in data_split[1]])

    return train, test, validate

def show_images(rec, gnd):
    # a = 1
    # ifr = sitk.ImageFileReader()
    # ifr.SetFileName('../data/10105/10105_19104511215149791073699219583840411343_needle_0.mha')
    # image = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')

    # rec = from_tensor_format(rec.detach().cpu(), True)[0]
    gnd2 = from_tensor_format(gnd.detach().cpu(), True).numpy()[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rec)
    ax1.set_title('Reconstruction')
    ax2.imshow(gnd2[0])
    ax2.set_title('Ground truth')
    ax3.imshow(gnd2[0] - rec)
    ax3.set_title('Difference')
    plt.show()