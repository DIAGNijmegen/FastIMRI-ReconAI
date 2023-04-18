import pytest

import numpy as np
from reconai.utils.metric import mse, ssim


@pytest.mark.usefixtures("data_batcher")
def test_ssim(data_batcher):

    img_generator = data_batcher.generate()
    slideshow = next(img_generator)
    image1 = slideshow[0, 0, :, :]

    assert ssim(image1, image1) == 1

    # Add more checks
    # image2 = image1.copy()
    # image2 = np.flip(image2)
    # score2 = ssim(image1, image2)


@pytest.mark.usefixtures("data_batcher")
def test_mse(data_batcher):
    img_generator = data_batcher.generate()
    slideshow = next(img_generator)
    image1 = slideshow[0, 0, :, :]

    assert mse(image1, image1) == 0

    assert np.isclose(mse(image1, image1 - 0.1), 0.01)


@pytest.mark.usefixtures("data_batcher")
def test_psnr(data_batcher):
    assert True