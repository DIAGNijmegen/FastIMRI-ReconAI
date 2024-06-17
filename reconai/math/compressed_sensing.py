import numpy as np
from . import fourier
from numpy.lib.stride_tricks import as_strided

from reconai.random import rng


def soft_thresh(u, lmda):
    """Soft-threshing operator for complex valued input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling)"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = rng().binomial(1, strided_pdf)

    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = rng().choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = fourier.ifftshift(mask, axes=(-1, -2))

    return mask


def shear_grid_mask(shape, acceleration_rate: int, sample_low_freq=True,
                    centred=False, sample_n=10):
    """
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    """
    Nt, Nx, Ny = shape
    start = rng().integers(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t) % acceleration_rate::acceleration_rate] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep


def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                              centred=False,
                              sample_n=10):
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t) % acceleration_rate::acceleration_rate] = 1

    # brute force
    rand_code = np.random.randint(0, 3, size=Nt*Nx)
    shift = np.array([-1, 0, 1])[rand_code]
    new_mask = np.zeros_like(mask)
    for t in range(Nt):
        for x in range(Nx):
            if mask[t, x]:
                new_mask[t, (x + shift[t*x]) % Nx] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        new_mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        new_mask[:, :xl] = 1
        new_mask[:, -xh:] = 1
    mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)

    return mask_rep


def undersample(x, mask, centred=False, norm='ortho', noise=0):
    """
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    x_u: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    """
    assert x.shape == mask.shape
    # zero mean complex Gaussian noise
    if noise > 0:
        noise_power = noise
        nz = np.sqrt(.5)*(rng().normal(0, 1, x.shape) + 1j * rng().normal(0, 1, x.shape))
        nz = nz * np.sqrt(noise_power)

        if norm == 'ortho':
            # multiplicative factor
            nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
        else:
            nz = nz * np.prod(mask.shape[-2:])
    else:
        nz = np.zeros(shape=x.shape, dtype=x.dtype)

    if centred:
        x_f = fourier.fft2c(x)
        x_fu = mask * (x_f + nz)
        x_u = fourier.ifft2c(x_fu)
        return x_u, x_fu
    else:
        x_f = fourier.fft2(x)
        x_fu = mask * (x_f + nz)
        x_u = fourier.ifft2(x_fu)
        return x_u, x_fu


def data_consistency(x, y, mask, centered=False, norm='ortho'):
    """
    x is in image space,
    y is in k-space
    """
    if centered:
        xf = fourier.fft2c(x)
        xm = (1 - mask) * xf + y
        xd = fourier.ifft2c(xm)
    else:
        xf = fourier.fft2(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = fourier.ifft2(xm, norm=norm)

    return xd


def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase


def undersampling_rate(mask):
    return float(mask.sum()) / mask.size
