import pytest
import torch
import numpy as np

from reconai.model.kspace_pytorch import DataConsistencyInKspace
from reconai.data.data import prepare_input


@pytest.mark.usefixtures("batcher")
def test_train(batcher):
    for im in batcher.items():
        im_und, k_und, mask, im_gnd = prepare_input(im, 1, 1)

        pass