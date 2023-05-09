import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def psnr(x, y):
    """
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    """
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.floating) \
        and np.issubdtype(y.dtype, np.floating)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    """
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    """
    mse_val = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse_val)
    else:
        return 10*np.log10(1./mse_val)

def ssim(gt, pred, transpose=True):
    """ Compute Structural Similarity Index Metric (SSIM). """
    if len(gt.shape) == 2:
        return compare_ssim(gt, pred, data_range=gt.max() - gt.min())
    elif len(gt.shape) == 3:
        if transpose:
            gt = gt.transpose(1, 2, 0)
            pred = pred.transpose(1, 2, 0)
        return compare_ssim(gt, pred, channel_axis=2, data_range=gt.max() - gt.min())
    else:
        raise NotImplementedError("Only for 2D and 3D data")

def ssim_2(img1, img2):
    def calc_ssim(imga, imgb):
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2
        imga = np.abs(imga).astype(np.float64)
        imgb = np.abs(imgb).astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(imga, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(imgb, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(imga ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(imgb ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(imga * imgb, -1, window)[5:-5, 5:-5] - mu1_mu2

        nom = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = nom / denom
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return calc_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(calc_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

