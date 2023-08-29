import pytest
import torch
import numpy as np

from reconai.model.kspace_pytorch import DataConsistencyInKspace
from reconai.data.data import prepare_input


@pytest.mark.usefixtures("batcher")
def test_dc_image_shifting(batcher):
    """
    Tests whether the images remain in the correct order after applying data consistency,
    when undersampling rate 1 is used. It compares mean and std of both images.
    """
    dc_layer = DataConsistencyInKspace()

    for im in batcher.items():
        im_und, k_und, mask, im_gnd = prepare_input(im, 1, 1)
        k_und = torch.complex(k_und[:, 0, ...], k_und[:, 1, ...]).unsqueeze(0)

        im_dc = dc_layer(im_und, k_und, mask).cpu()

        assert im_dc.shape == im_und.shape

        # Make sure the images are not shuffled
        for i in range(im_dc.shape[-1]):
            im_dc_i = im_dc[0, 0, :, :, i].numpy()
            im_und_i = im_und[0, 0, :, :, i].numpy()

            assert np.isclose(im_dc_i.mean(), im_und_i.mean())
            assert np.isclose(np.std(im_dc_i), np.std(im_und_i))
