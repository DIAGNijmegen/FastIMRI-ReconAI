import json
from pathlib import Path

from reconai.data.data import gather_data
from reconai.data.Batcher import Batcher as OldBatcher

import pytest
import numpy as np

from reconai.data.batcher1 import Batcher
from reconai.data.dataloader import DataLoader
from reconai.data.sequence import SequenceCollection
from reconai.data.sequencer import Sequencer


@pytest.fixture
def dataloader():
    dl = DataLoader(Path('./input'))
    dl.load('.*_(.*)_')
    return dl


def test_volume():
    data = gather_data(Path('input'))
    data_error = OldBatcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    # data_n = len(data)

    batcher = OldBatcher(data)
    for item in batcher.generate():
        # TODO: check this with input yaml config file rather than constant val
        assert item.shape == (1, 15, 256, 256)
    pass


@pytest.fixture
def sequences(dataloader: DataLoader):
    seq = Sequencer(dataloader)
    obj = seq.generate_sequences(seed=10, seq_len=5, mean_slices_per_mha=2, max_slices_per_mha=3, q=0.5)
    assert len(obj) == 12, 'input data has changed'
    return obj


def test_generate_sequence(dataloader: DataLoader, sequences: SequenceCollection):
    # with open('./output/test_data_expected_sequences.json', 'w') as f:
    #     json.dump(repr(sequences), f, indent=1)
    with open('./output/test_data_expected_sequences.json') as f:
        assert json.load(f) == repr(sequences)

    seq = Sequencer(dataloader)
    kwargs = {'seed': 11, 'seq_len': 5, 'mean_slices_per_mha': 2, 'max_slices_per_mha': 3, 'q': 0.5}
    obj1 = seq.generate_sequences(**kwargs)
    obj2 = seq.generate_sequences(**kwargs)
    assert obj1 == obj2
    kwargs['seed'] = 0
    assert obj1 != seq.generate_sequences(**kwargs)


def test_batcher(dataloader: DataLoader, sequences: SequenceCollection):
    batcher = Batcher(dataloader)
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
