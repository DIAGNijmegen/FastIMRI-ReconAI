import pytest
import torch
from pathlib import Path
import numpy as np

from reconai.cascadenet_pytorch.kspace_pytorch import DataConsistencyInKspace
from reconai.data.data import gather_data, prepare_input
from reconai.data.Batcher import Batcher


@pytest.fixture
def data_batcher() -> Batcher:
    data = gather_data(Path('input'))
    data_error = Batcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    return Batcher(data)

def test_dataconsistency(data_batcher):
    """
    Tests whether the images remain in the correct order after applying data consistency,
    when undersampling rate 1 is used. It compares mean and std of both images.
    """
    dc_layer = DataConsistencyInKspace()

    for im in data_batcher.generate():
        im_und, k_und, mask, im_gnd = prepare_input(im, 1)
        k_und = torch.complex(k_und[:, 0, ...], k_und[:, 1, ...]).unsqueeze(0)

        im_dc = dc_layer(im_und, k_und, mask)
        im_dc = im_dc.cpu()

        assert im_dc.shape == im_und.shape

        # Make sure the images are not shuffled
        for i in range(im_dc.shape[-1]):
            im_dc_i = im_dc[0, 0, :, :, i].numpy()
            im_und_i = im_und[0, 0, :, :, i].numpy()

            assert np.isclose(im_dc_i.mean(), im_und_i.mean())
            assert np.isclose(np.std(im_dc_i), np.std(im_und_i))





