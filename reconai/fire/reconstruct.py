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
from ..data import preprocess_real, image


class FireReconstruct(FireModule):
    def __init__(self):
        super().__init__()
        self._params: ModelParameters | None = None
        self._network: CRNNMRI | None = None
        self._export: np.ndarray = np.ndarray(0, dtype=np.float32)
        self._export_suffix: int = 0
        self._debug = False

    def load(self, **kwargs):
        self._debug = kwargs.get('debug', False)

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

    def _yield_export(self, data: np.ndarray, suffix: str = None):
        self._export = data
        self._export_suffix = suffix
        return self._export

    def run(self, data: np.ndarray, metadata: dict) -> None | np.ndarray | dict | tuple[np.ndarray, dict]:
        """
        Accepts data of shape rep, y, x of dtype complex64, yields data of shape y, x of dtype float32

        This undersamples such that the undersampling is equal to the model's trained undersampling rate.
        """
        data = data.astype(np.complex128)
        rep, y, x = data.shape
        params_shape = (y, y)

        sampled_rows = np.abs(data[0, :, 0]) > 0
        actual_rows = sampled_rows.sum()
        target_rows = y / self._params.data.undersampling
        mask_rows = np.ones(shape=(rep, y), dtype=bool)
        if actual_rows > target_rows:
            mask_sampled = [i for i, b in enumerate(sampled_rows) if b]
            for t in range(rep):
                mask = get_rand_exp_decay_mask(actual_rows, 1, target_rows / actual_rows, 1 / 3)
                mask_m2i = mask.flatten() * mask_sampled
                mask_m2i = mask_m2i[mask_m2i != 0]
                if np.all(mask[0]):
                    mask_m2i = np.pad(mask_m2i, (1, 0))
                mask_rows[t, mask_m2i.astype(np.integer)] = False
                data[t, mask_rows[t]] = 0
        mask_rows = 1 - np.repeat(mask_rows.astype(np.integer)[:, :, np.newaxis], y, axis=2)

        # warn if data.shape[1] != self._params.data.shape_y
        image = np.ndarray(shape=(rep, *params_shape), dtype=data.dtype)
        phase = np.empty_like(image)
        k = np.empty_like(image)
        cy, cx = [(a - b) // 2 for a, b in zip((y, x), params_shape)]

        for t in range(rep):
            phase[t] = ifft2c(data[t])[cy:-cy if cy > 0 else None, cx:-cx if cx > 0 else None]
            # phase[t] = np.rot90(phase[t], 2)
            image[t] = np.abs(phase[t])
            if self._debug and t < 5:
                yield self._yield_export(image[t], f'{t}_ifft_abs')
            image[t] = np.clip(np.divide(image[t], np.percentile(image[t], 99)), 0, 1)
            if self._debug and t < 5:
                yield self._yield_export(image[t], f'{t}_ifft_abs_norm')

        for t in range(rep):
            if self._debug and t < 5:
                yield self._yield_export(image[t], f'{t}_ifft_abs_norm')
            k[t] = fft2c(image[t] * np.exp(1j * np.angle(phase[t])))
            if self._debug and t < 5:
                yield self._yield_export(ifft2c(k[t]), f'{t}_ifft_abs_norm_from_k')

        image = np.expand_dims(image.astype(np.float32), axis=0)
        mask_rows = np.expand_dims(mask_rows, axis=0)
        k = np.expand_dims(k, axis=0)
        win, dow = 0, self._params.data.sequence_length
        with torch.no_grad():
            while dow <= data.shape[1]:
                imageT, kT, maskT = preprocess_real(image[:, win:dow], k[:, win:dow], mask_rows[:, win:dow])

                if self._debug:
                    if win >= 5:
                        break
                    yield self._yield_export(imageT[0, 0, :, :, -1].cpu().numpy(), f'{win}_pred_input')

                pred, _ = self._network(imageT, kT, maskT, test=True)

                yield self._yield_export(pred[0, 0, :, :, -1].cpu().numpy(), f'{win}_pred_output')

                win += 1
                dow += 1
        yield None

    def export(self, export_dir: Path):
        suffix = f'.{self._export_suffix}' if self._export_suffix else ''
        cv2.imwrite((export_dir / f'{self._params.meta.name}{suffix}.png').as_posix(), image(self._export))

    @staticmethod
    def _normal(array: np.ndarray, norm: float, maximum_1: bool = True) -> np.ndarray:
        norm = np.zeros(array.shape) + norm
        if maximum_1:
            return np.clip(np.divide(array, norm), 0, 1)
        return np.divide(array, norm)
