from typing import List, Callable
from datetime import datetime

import numpy as np
import torch
from piqa import SSIM

from .parameters import Parameters


class Evaluation:
    class Criterion:
        def __init__(self, crit: Callable):
            # add loss_weight=None, name:str to simplify Evaluation by a lot
            self._crit = crit
            self._n = 0
            self._value = None

        def calculate(self, pred: np.ndarray, gnd: np.ndarray) -> torch.Tensor:
            if not self._value:
                self._value = torch.tensor(0, device='cuda', dtype=gnd.dtype)
            return self._crit(pred, gnd)

        def add(self, value: torch.Tensor):
            self._n += 1
            self._value += value

        @property
        def value(self):
            return (self._value / self._n) if self._n > 0 else 0

    def __init__(self, params: Parameters, loss_only: bool = False):
        self._crit: dict[str, Evaluation.Criterion] = {
            'mse': Evaluation.Criterion(torch.nn.MSELoss().cuda()),
            'ssim': Evaluation.Criterion(SSIM(n_channels=params.data.sequence_length).cuda()),
            'dice': Evaluation.Criterion(lambda pred, gnd: torch.tensor(0, device='cuda', dtype=gnd.dtype))
        }
        self._weights: dict[str, float] = {
            'mse': params.train.loss.mse,
            'ssim': params.train.loss.ssim,
            'dice': params.train.loss.dice
        }

        self._last_calcs: dict[str, torch.Tensor] = {key: None for key in self._crit.keys()}
        self._loss_only = loss_only
        self._loss = Evaluation.Criterion(lambda pred, gnd: self._weighted_loss(pred, gnd))

        self._time = Evaluation.Criterion(lambda pred, gnd: torch.tensor((datetime.now() - self.__time).microseconds))
        self.__time = datetime.now()

        self._paths: dict[str, dict[str, float]] = {}

    def start_timer(self):
        self.__time = datetime.now()

    def calculate(self, pred, gnd, path: str = None) -> torch.Tensor:
        """
        Calculate all criterions (MSE, SSIM and Dice), but returns only the loss tensor.
        """
        for key, crit in self._crit.items():
            if self._loss_only and self._weights[key] == 0:
                continue

            if key == 'dice':
                continue  # NYI
            elif key == 'ssim':
                pred_permute, gnd_permute = pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0]
                self._last_calcs[key] = self._crit[key].calculate(pred_permute, gnd_permute)
            else:
                self._last_calcs[key] = self._crit[key].calculate(pred, gnd)
            crit.add(self._last_calcs[key])

        self._loss.add(loss := self._loss.calculate(pred, gnd))
        self._time.add(time := self._time.calculate(pred, gnd))

        if path:
            self._paths[path] = ({key: self._last_calcs[key].item() for key in self._crit.keys()} |
                                 {'loss': loss.item(), 'time': time.item()})

        return loss

    def keys(self):
        return self._crit.keys()

    @property
    def paths(self) -> dict[str, dict[str, float]]:
        return self._paths

    def __getitem__(self, item: str):
        if self._loss_only and item != 'loss':
            raise KeyError()
        return self._crit[item].value

    def _calculate(self, key: str, pred, gnd):
        if key == 'ssim':
            pred_permute, gnd_permute = pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0]
            return self._crit[key].calculate(pred_permute, gnd_permute)
        else:
            return self._crit[key].calculate(pred, gnd)

    def _weighted_loss(self, pred, gnd) -> torch.Tensor:
        loss_sum = torch.tensor(0, device='cuda', dtype=gnd.dtype)
        for key in self._crit.keys():
            if self._weights.get(key, 0) > 0:
                if key == 'mse':
                    loss_sum += self._weights[key] * self._last_calcs[key]
                else:
                    loss_sum += self._weights[key] * (1 - self._last_calcs[key])
        return loss_sum
