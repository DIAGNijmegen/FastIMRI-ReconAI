import torch
import torch.fft as fourier

from reconai.cascadenet_pytorch.module import Module


class DataConsistencyInKspace(Module):
    """ Create data consistency operator in kspace

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm
        self.noise_lvl = noise_lvl

    def forward(self, *input_params, **kwargs):
        return self.perform(*input_params)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, n_ch, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        batch, n_ch, height, width, sequence_length = x.shape
        if batch != 1 or n_ch != 1:
            raise NotImplementedError("Only implemented batchsize 1 and n_ch 1")

        x = x.permute(0, 1, 4, 2, 3).squeeze()
        k0 = k0.permute(0, 1, 4, 2, 3).squeeze()
        mask = mask.permute(0, 1, 4, 2, 3).squeeze()

        x_res = torch.empty(size=(sequence_length, height, width), dtype=torch.float32, device="cuda")
        for i in range(sequence_length):
            k = fourier.fftshift(fourier.fft2(fourier.ifftshift(x[i]), norm=self.normalized))
            k_c = self.data_consistency(k, k0[i], mask[i], self.noise_lvl)
            x_res[i, :, :] = torch.abs(fourier.fftshift(fourier.ifft2(fourier.ifftshift(k_c), norm=self.normalized)))

        x_res = x_res[None, None, :, :, :]  # Add 2 dims back again
        x_res = x_res.permute(0, 1, 3, 4, 2)
        return x_res

        # OLD
        # n_ch = x.shape[1]
        # if x.dim() == 4:  # input is 2D
        #     x = x.permute(0, 2, 3, 1)
        #     k0 = k0.permute(0, 2, 3, 1)
        #     mask = mask.permute(0, 2, 3, 1)
        # elif x.dim() == 5:  # input is 3D
        #     x = x.permute(0, 4, 2, 3, 1)
        #     k0 = k0.permute(0, 4, 2, 3, 1)
        #     mask = mask.permute(0, 4, 2, 3, 1)
        #
        # if n_ch == 1:
        #     dim = (2, 3)
        #     k = fourier.fft2(x, dim=dim)
        #     k_c = self.data_consistency(k, k0, mask, self.noise_lvl)
        #     x_res = torch.abs(fourier.ifftshift(fourier.ifft2(k_c, dim=dim), dim=dim))
        # else:
        #     raise NotImplementedError()

        # if x.dim() == 4:
        #     x_res = x_res.permute(0, 3, 1, 2)
        # elif x.dim() == 5:
        #     x_res = x_res.permute(0, 4, 2, 3, 1)
        #
        # return x_res

    @staticmethod
    def data_consistency(k, k0, mask, noise_lvl=None):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        noise_lvl - noise
        """

        if noise_lvl:  # noisy case
            return (1 - mask) * k + mask * (k + noise_lvl * k0) / (1 + noise_lvl)
        else:  # noiseless case
            return (1 - mask) * k + mask * k0
