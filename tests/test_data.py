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
    b.load()
    pass
    # output_mha = Volume.load(Path('./output/1_3_sag000.mha'))
    #
    # data = gather_data(Path('input'))
    # Batcher.shuffle = False
    # batcher = Batcher(data)
    # for item in batcher.generate():
    #     # TODO: check this with input yaml config file rather than constant val
    #     assert item.shape == (1, 15, 256, 256)
    # pass