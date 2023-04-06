import json
from pathlib import Path

import pytest
import numpy as np

from reconai.data.Batcher import Batcher1, DataLoader, SequenceCollection


@pytest.fixture
def dataloader():
    dl = DataLoader(Path('./input'))
    dl.load('.*_(.*)_')
    return dl


@pytest.fixture
def sequences(dataloader: DataLoader):
    obj = dataloader.generate_sequences_from_dataset(seed=10, seq_len=5, mean_slices_per_mha=2, max_slices_per_mha=3, q=0.5)
    assert len(obj) == 12, 'input data has changed'
    return obj


def test_generate_sequence(dataloader: DataLoader, sequences: SequenceCollection):
    # with open('./output/test_data_expected_sequences.json', 'w') as f:
    #     json.dump(repr(sequences), f, indent=1)
    with open('./output/test_data_expected_sequences.json') as f:
        assert json.load(f) == repr(sequences)

    kwargs = {'seed': 11, 'seq_len': 5, 'mean_slices_per_mha': 2, 'max_slices_per_mha': 3, 'q': 0.5}
    obj1 = dataloader.generate_sequences_from_dataset(**kwargs)
    obj2 = dataloader.generate_sequences_from_dataset(**kwargs)
    assert obj1 == obj2
    kwargs['seed'] = 0
    assert obj1 != dataloader.generate_sequences_from_dataset(**kwargs)


def test_batcher(dataloader: DataLoader, sequences: SequenceCollection):
    batcher = Batcher1(dataloader)
    for s in sequences.items():
        batcher.append_sequence(s)

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

    batcher.shuffle(10)
    obj1 = next(batcher.items())
    batcher.shuffle(10)
    obj2 = next(batcher.items())
    assert np.all(obj1 == obj2)
    batcher.shuffle(9)
    obj2 = next(batcher.items())
    assert not np.all(obj1 == obj2)

