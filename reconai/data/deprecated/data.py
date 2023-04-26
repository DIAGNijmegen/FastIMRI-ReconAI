import logging
import random
from pathlib import Path
from typing import List

import numpy as np
from box import Box

from reconai.data.deprecated.Batcher import Batcher
from reconai.data.deprecated.Volume import Volume


def gather_data(data_dir: Path, debug: bool = False):
    data = []
    for patient_dir in data_dir.iterdir():
        try:
            if patient_dir.is_dir():
                files = list(patient_dir.iterdir())
                study_ids = {fn.name.split('_')[1] for fn in files if not fn.name.startswith('tmp')}
                for study_id in study_ids:
                    data.append(Volume(study_id, [fn for fn in files if study_id in fn.name]))
        except:
            continue
        if debug and len(data) > 20:
            break
    return data


def get_data_volumes(args: Box) -> List[Volume]:
    Batcher.batch_size = args.batch_size
    Volume.sequence_length = args.sequence_len

    data = gather_data(args.in_dir, args.debug)
    data_error = Batcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    data_n = len(data)
    logging.info(f"{data_n} volumes found, {len(data_error)} dropped out")
    if data_n < 3:
        raise ValueError('insufficient data for training')

    return data


def get_dataset_batchers(args: Box, data_volumes: List[Volume], n_folds: int, fold: int) -> (Batcher, Batcher, Batcher):
    data_n = list(range(len(data_volumes)))
    # random.seed(args.seed)
    random.seed(5)
    random.shuffle(data_n)
    data_split = np.array_split(data_n, n_folds + 1 if n_folds > 2 else 5)

    k_validation = {fold + 1}
    k_training = set(range(1, len(data_split))).difference(k_validation)
    # k_test is 0

    train = Batcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_training])])
    validate = Batcher([data_volumes[i] for i in np.concatenate([data_split[i] for i in k_validation])])
    test = Batcher([data_volumes[i] for i in data_split[0]])

    return train, validate, test


# def show_images(rec, gnd):
#     # a = 1
#     # ifr = sitk.ImageFileReader()
#     # ifr.SetFileName('../data/10105/10105_19104511215149791073699219583840411343_sag_0.mha')
#     # image = sitk.GetArrayFromImage(ifr.Execute()).astype('float64')
#
#     # rec = from_tensor_format(rec.detach().cpu(), True)[0]
#     gnd2 = from_tensor_format(gnd.detach().cpu(), True).numpy()[0]
#
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     ax1.imshow(rec)
#     ax1.set_title('Reconstruction')
#     ax2.imshow(gnd2[0])
#     ax2.set_title('Ground truth')
#     ax3.imshow(gnd2[0] - rec)
#     ax3.set_title('Difference')
#     plt.show()


def get_data_information(args):
    # Volume.key = 'needle'
    data = get_data_volumes(args)
    train, validate, _ = get_dataset_batchers(args, data, 1, 0)

    mins = []
    maxes = []
    max_to_perc99 = []
    averages = []
    for image in train.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            mins.append(slice.min())
            maxes.append(slice.max())
            averages.append(slice.mean())
            perc99 = np.percentile(slice, 99)
            max_to_perc99.append(slice.max() - perc99)

    for image in validate.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            maxes.append(slice.max())

    # print(f'Min {min(mins)}')
    print(f'Min of maxes {min(maxes)}')
    print(f'Average of maxes {np.mean(maxes)}')
    print(f'std of maxes {np.std(maxes)}')
    print(f'Max of maxes {max(maxes)}')
    # print(f'Avg {np.mean(averages)}')
    # print(f'Min diff max-perc99 {np.min(max_to_perc99)}')
    # print(f'Avg diff max-perc99 {np.mean(max_to_perc99)}')
    # print(f'Max diff max-perc99 {np.max(max_to_perc99)}')
