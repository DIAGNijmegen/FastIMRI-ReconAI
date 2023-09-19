from typing import List

import torch
from piqa import SSIM

from .parameters import Parameters


class Criterion:
    def __init__(self, params: Parameters):
        self._mse = params.train.loss.mse
        self._ssim = params.train.loss.ssim
        self._dice = params.train.loss.dice

        self._crit = {
            'mse': torch.nn.MSELoss().cuda(),
            'ssim': SSIM(n_channels=params.data.sequence_length).cuda(),
            'dice': lambda: 0
        }

    def mse(self, pred, gnd):
        return self._crit['mse'](pred, gnd)

    def ssim(self, pred, gnd):
        pred_permute, gnd_permute = pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0]
        return self._crit['ssim'](pred_permute, gnd_permute)

    def dice(self, pred, gnd):
        return 0

    def weighted_loss(self, pred, gnd) -> torch.Tensor:
        weighted_loss: List[(float, torch.Tensor)] = []
        if self._mse > 0:
            weighted_loss.append((self._mse, self.mse(pred, gnd)))
        if self._ssim > 0:
            weighted_loss.append((self._ssim, 1 - self.ssim(pred, gnd)))
        if self._dice > 0:
            raise NotImplementedError("Only MSE or SSIM loss implemented")

        loss_sum = torch.tensor(0, device='cuda', dtype=gnd.dtype)
        for weight, value in weighted_loss:
            loss_sum += weight * value
        return loss_sum
