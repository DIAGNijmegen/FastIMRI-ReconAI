import json
from pathlib import Path

import pytest
import numpy as np
import SimpleITK as sitk

from reconai.data.data import gather_data
from reconai.data.Batcher import Batcher, Batcher1
from reconai.data.Volume import Volume


# @pytest.fixture
# def itk():
#     return sitk.ImageFileReader()


def test_volume():
    b = Batcher1(Path('./input'))
    b.load('.*_(.*)_')
    b.prepare_sequences(seed=10, seq_len=15, mean_slices_per_mha=2, max_slices_per_mha=3, q=0.5)

    with open('./output/test_data_expected_sequences.json') as f:
        assert json.load(f) == b._sequences