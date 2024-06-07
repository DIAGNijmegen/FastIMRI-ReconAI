from pathlib import Path

import numpy as np
import torch

from fire.module import FireModule
from ..parameters import ModelParameters
from ..reconstruction import CRNNMRI
from ..random import rng
from ..math.kspace import get_rand_exp_decay_mask
from ..data import preprocess_as_variable


class FireReconstruct(FireModule):
    def __init__(self):
        super().__init__()
        self._params: ModelParameters | None = None
        self._network: CRNNMRI | None = None

    def load(self, **kwargs):
        model_dir = Path(kwargs['model_dir'])
        params = ModelParameters(model_dir, model_dir)
        rng(params.data.seed)

        self.logger.info(f'Loaded ReconAI model: {params.meta.name}')

        self._network = CRNNMRI(n_ch=params.model.channels,
                          nf=params.model.filters,
                          ks=params.model.kernelsize,
                          nc=params.model.iterations,
                          nd=params.model.layers,
                          bcrnn=params.model.bcrnn
                          ).cuda()

        self._network.load_state_dict(torch.load(params.npz))
        self._network.eval()
        self._params = params

    def run(self, data: np.ndarray, metadata: dict) -> None | np.ndarray | dict | tuple[np.ndarray, dict]:
        sampled_rows = np.abs(data[0, :, 0]) > 0
        target_rows = data.shape[1] / self._params.data.undersampling
        actual_rows = sampled_rows.sum()
        if actual_rows > target_rows:
            m2i = [i for i, b in enumerate(sampled_rows) if b]
            for t in range(data.shape[0]):
                mask = get_rand_exp_decay_mask(actual_rows, 1, target_rows / actual_rows, 1 / 3)
                m2i_mask = (1 - mask.flatten()) * m2i
                m2i_mask[0] = 1 if np.all(mask[0] == 0) else 0
                m2i_mask: np.ndarray = m2i_mask[m2i_mask > 0]
                data[t, m2i_mask.astype(np.integer)] = 0

        params_shape = (self._params.data.shape_y, self._params.data.shape_y)
        if params_shape != data.shape[1:]:
            data_ = np.ndarray(shape=(data.shape[0], *params_shape), dtype=data.dtype)
            crop = [(a - b) // 2 for a, b in zip(data.shape[1:], params_shape)]
            for t in range(data.shape[0]):
                i = np.fft.ifftshift(np.fft.ifft2(data[t]))
                i = i[crop[0]:crop[0] + params_shape[0], crop[1]:crop[1] + params_shape[1]]
                data_[t] = np.fft.fft2(np.fft.fftshift(i))
            data = data_

        result = np.ndarray(shape=(data.shape[0], *params_shape))
        with torch.no_grad():
            data = np.expand_dims(data, axis=0)
            win, dow = 0, self._params.data.sequence_length
            while dow <= data.shape[1]:
                im, k, mask, _ = preprocess_as_variable(data[win:dow], acceleration=1)
                pred, _ = self._network(im, k, mask, test=True)
                win += 1
                dow += 1
            pass
            # datapiece = DataLoader(Dataset(tempdir, normalize=params.data.normalize, sequence_len=params.data.sequence_length))
            # for piece in datapiece:
            #     im_u, k_u, mask, _ = preprocess_as_variable(piece['data'], params.data.undersampling)
            #     for i in range(len(piece['paths'])):
            #         j = i + 1
            #         pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
            #
            #         sitk_image = sitk.GetImageFromArray(pred.squeeze(dim=(0, 1)).cpu().numpy().transpose(2, 0, 1))
            #         sitk_image.SetOrigin([float(o[i]) for o in piece['origin']])
            #         sitk_image.SetDirection([float(d[i]) for d in piece['direction']])
            #         sitk_image.SetSpacing([float(d[i]) for d in piece['spacing']])
            #         sitk.WriteImage(sitk_image, out.resolve())

        yield None

    def export(self, export_dir: Path):
        pass
