import json
from pathlib import Path

import pytest
import numpy as np
import SimpleITK as sitk

from reconai.data.data import gather_data
from reconai.data.Batcher import Batcher, Batcher1, DataLoader, Sequences
from reconai.data.Volume import Volume


@pytest.fixture
def dataloader():
    dl = DataLoader(Path('./input'))
    dl.load('.*_(.*)_')
    return dl


@pytest.fixture
def sequences(dataloader: DataLoader):
    return dataloader.generate_sequences(seed=10, seq_len=15, mean_slices_per_mha=2, max_slices_per_mha=3, q=0.5)


def test_generate_sequence(sequences: Sequences):
    with open('./output/test_data_expected_sequences.json') as f:
        assert json.load(f) == sequences


def test_append_sequence(dataloader: DataLoader, sequences: Sequences):
    batcher = Batcher1(dataloader)
    batcher.append_sequences(sequences)