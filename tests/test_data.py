import json
from pathlib import Path
import matplotlib.pyplot as plt

import pytest
import numpy as np

from reconai.data.batcher import Batcher, zoom, rotate, crop_or_pad, flip_ud_lr, normalize
from reconai.data.dataloader import DataLoader
from reconai.data.sequence import SequenceCollection
from reconai.data.sequencebuilder import SequenceBuilder


@pytest.fixture
def dataloader() -> DataLoader:
    dl = DataLoader(Path('./input'))
    dl.load('.*_(.*)_')
    return dl


@pytest.fixture
def sequences(dataloader: DataLoader) -> SequenceCollection:
    seq = SequenceBuilder(dataloader)
    obj = seq.generate_multislice_sequences(seed=10, seq_len=5, mean_slices_per_mha=2, max_slices_per_mha=3, q=0.5)
    assert len(obj) == 12, 'input data has changed'
    return obj


@pytest.fixture
def batcher(dataloader, sequences):
    a_batcher = Batcher(dataloader)

    for sequence in sequences.items():
        a_batcher.append_sequence(sequence=sequence,
                                  crop_expand_to=(256, 256),
                                  norm=1961.06,
                                  equal_images=False)
    return a_batcher


def test_generate_multislice_sequences(dataloader: DataLoader, sequences: SequenceCollection):
    # with open('./output/test_data_expected_sequences.json', 'w') as f:
    #     json.dump(repr(sequences), f, indent=1)
    with open('./output/test_data_expected_sequences.json') as f:
        assert json.load(f) == repr(sequences)

    seq = SequenceBuilder(dataloader)
    kwargs = {'seed': 11, 'seq_len': 5, 'mean_slices_per_mha': 2, 'max_slices_per_mha': 3, 'q': 0.5}
    obj1 = seq.generate_multislice_sequences(**kwargs)
    obj2 = seq.generate_multislice_sequences(**kwargs)
    assert obj1 == obj2
    kwargs['seed'] = 0
    assert obj1 != seq.generate_multislice_sequences(**kwargs)


def test_generate_sequences(dataloader: DataLoader):
    seq = SequenceBuilder(dataloader)
    obj = seq.generate_sequences(seed=10, seq_len=5)
    obj_random1 = seq.generate_sequences(seed=10, seq_len=5, random_order=True)
    obj_random2 = seq.generate_sequences(seed=10, seq_len=5, random_order=True)
    assert obj_random1 == obj_random2
    obj_1 = seq.generate_sequences(seed=10, seq_len=1)
    assert all([list(x.items())[0][1][0] == 2 for x in obj.items()])
    obj_20 = seq.generate_sequences(seed=10, seq_len=20)
    assert np.all([list(x.items())[0][1] == [1, 3, 0, 4, 0, 3, 1, 2, 1, 3, 0, 4, 0, 3, 1, 2, 1, 3, 0] for x in obj_20.items()])

    for o in [obj, obj_random1, obj_random1, obj_1, obj_20]:
        assert len(o) == 12


def test_batcher_append_sequence(dataloader: DataLoader, sequences: SequenceCollection):
    batcher = Batcher(dataloader)
    batcher.append_sequence(next(sequences.items()))

    show = False
    expected_output = './output/test_batcher_append_sequence_expected_'

    def imshow(img, title, show=False, save=False):
        fig = plt.figure()
        plt.imshow(img, cmap='Greys_r')
        fig.suptitle(title)
        if save:
            np.save(expected_output + title, img)
        if show:
            fig.show()

    sequence_images = next(batcher.items())[0][3]
    mean = sequence_images.mean()
    imshow(sequence_images, "original", show=show)

    z = 2
    sequence_images_zoom = zoom(sequence_images, z)
    assert all(sequence_images.shape[i] * z == sequence_images_zoom.shape[i] for i in range(2))
    assert np.isclose(mean, sequence_images_zoom.mean(), atol=2)
    imshow(sequence_images_zoom, "zoom", show=show)

    sequence_images_rotate = rotate(sequence_images, 45)
    sequence_images_rotate_path = Path(expected_output + 'rotate').with_suffix('.npy')
    imshow(sequence_images_rotate, "rotate", show=show)
    # assert np.allclose(np.load(str(sequence_images_rotate_path)), sequence_images_rotate)

    for x in [128, 256, 512]:
        for y in [128, 256, 512]:
            sequence_images_crop = crop_or_pad(sequence_images, (x, y))
            assert sequence_images_crop.shape == (x, y)

    for f in [[''], ['lr'], ['ud'], ['lr', 'ud']]:
        sequence_images_flip = flip_ud_lr(sequence_images, *f)
        sequence_images_flip_path = Path(expected_output + 'flip' + ''.join(f)).with_suffix('.npy')
        imshow(sequence_images_flip, 'flip' + ''.join(f), show=show)
        assert np.isclose(mean, sequence_images_flip.mean())
        # assert np.allclose(np.load(str(sequence_images_flip_path)), sequence_images_flip)

    sequence_images_norm = normalize(sequence_images, 2000)
    assert np.isclose(mean / 2000, sequence_images_norm.mean())


def test_batcher(dataloader: DataLoader, sequences: SequenceCollection):
    batcher = Batcher(dataloader)
    for s in sequences.items():
        batcher.append_sequence(s, norm=2, flip='lr', zoom_factor=1.1, rotate_deg=5)

    max_folds = 4

    training, validation = {}, {}
    for fold in range(max_folds):
        training[fold] = [s for s in batcher.items_fold(fold, max_folds=max_folds)]
        validation[fold] = [s for s in batcher.items_fold(fold, max_folds=max_folds, validation=True)]

    full = list(batcher.items())
    for fold in range(max_folds):
        tr, va = training[fold], validation[fold]
        tr_va = tr + va
        assert len(tr_va) == len(full)
        # check whether tr + va == full
        for tv in tr_va:
            assert any(np.all(tv == f) for f in full)
        # check whether tr △ va
        for v in va:
            assert all(not np.array_equal(v, t) for t in tr)

    # test for seed
    batcher.shuffle(10)
    obj1 = next(batcher.items())
    batcher.shuffle(10)
    obj2 = next(batcher.items())
    assert np.all(obj1 == obj2)
    batcher.shuffle(9)
    obj2 = next(batcher.items())
    assert not np.all(obj1 == obj2)
