from pathlib import Path

import numpy as np
import torch
import cv2

from fire.module import FireModule
from ..parameters import ModelParameters
from ..reconstruction import CRNNMRI
from ..random import rng
from ..math.kspace import get_rand_exp_decay_mask
from ..math.fourier import fft2c, ifft2c
from ..data import preprocess_real, preprocess_simulated


class FireReconstruct(FireModule):
    def __init__(self):
        super().__init__()
        self._params: ModelParameters | None = None
        self._network: CRNNMRI | None = None
        self._export: np.ndarray = np.ndarray(0, dtype=np.float32)
        self._export_i: int = 0

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
        """
        Accepts data of shape rep, y, x of dtype complex64, yields data of shape y, x of dtype float32

        This undersamples such that the undersampling is equal to the model's trained undersampling rate.
        """
        data = data.astype(np.complex128)
        rep, y, x = data.shape

        sampled_rows = np.abs(data[0, :, 0]) > 0
        actual_rows = sampled_rows.sum()
        target_rows = y / self._params.data.undersampling
        if actual_rows > target_rows:
            m2i = [i for i, b in enumerate(sampled_rows) if b]
            for t in range(rep):
                mask = get_rand_exp_decay_mask(actual_rows, 1, target_rows / actual_rows, 1 / 3)
                m2i_mask = (1 - mask.flatten()) * m2i
                m2i_mask[0] = 1 if np.all(mask[0] == 0) else 0
                m2i_mask: np.ndarray = m2i_mask[m2i_mask > 0]
                data[t, m2i_mask.astype(np.integer)] = 0

        params_shape = (y, y)
        # warn if data.shape[1] != self._params.data.shape_y
        image = np.ndarray(shape=(rep, *params_shape), dtype=data.dtype)
        k = np.ndarray(shape=(rep, *params_shape), dtype=data.dtype)
        cy, cx = [(a - b) // 2 for a, b in zip((y, x), params_shape)]
        norm = 0
        
        for t in range(rep):
            image[t] = ifft2c(data[t, cy:-cy if cy > 0 else None, cx:-cx if cx > 0 else None])
            norm += np.percentile(image[t], 99) / rep
        image = np.clip(np.divide(image, norm), 0, 1)
        for t in range(rep):
            k[t] = fft2c(image[t])

        image = np.expand_dims(np.abs(image).astype(np.float32), axis=0)
        k = np.expand_dims(k, axis=0)
        win, dow = 0, self._params.data.sequence_length
        with torch.no_grad():
            while dow <= data.shape[1] and win < 5:
                preprocess_simulated(image[:, win:dow], acceleration=1)
                imageT, kT, maskT = preprocess_real(image[:, win:dow], k[:, win:dow], sampled_rows[cy:-cy if cy > 0 else None])

                pred, _ = self._network(imageT, kT, maskT, test=True)

                self._export = pred[0, 0, :, :, -1].cpu().numpy()
                self._export_i = win
                yield self._export

                win += 1
                dow += 1
        yield None

    def export(self, export_dir: Path):
        array_min, array_max = self._export.min(), self._export.max() + np.finfo(self._export.dtype).eps
        array_norm = (self._export - array_min) / (array_max - array_min)
        array_uint8 = (np.clip(array_norm, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite((export_dir / f'{self._params.meta.name}.{self._export_i}.png').as_posix(), array_uint8)

    @staticmethod
    def _normal(array: np.ndarray, norm: float, maximum_1: bool = True) -> np.ndarray:
        norm = np.zeros(array.shape) + norm
        if maximum_1:
            return np.clip(np.divide(array, norm), 0, 1)
        return np.divide(array, norm)
