from typing import List, Callable
from datetime import datetime

import numpy as np
import torch
from piqa import SSIM

from .parameters import Parameters


def tensor_zero(dtype: torch.dtype):
    return torch.tensor(0, device='cuda', dtype=dtype)


class Evaluation:
    class Criterion:
        def __init__(self, name: str, criterion: Callable, loss_weight: float | None = None):
            # add loss_weight=None, name:str to simplify Evaluation by a lot
            self._name = name
            self._loss_weight = loss_weight
            self._criterion = criterion
            self._n = 0
            self._value = torch.tensor(0)
            self._result = torch.tensor(0)

        def calculate(self, pred: np.ndarray, gnd: np.ndarray):
            if not self._value:
                self._value = tensor_zero(gnd.dtype)

            self._result: torch.Tensor = self._criterion(pred, gnd)
            self._n += 1
            self._value += self._result

        @property
        def result(self) -> torch.Tensor:
            return self._result

        @property
        def loss_weight(self) -> float | None:
            return self._loss_weight

        @property
        def name(self) -> str:
            return self._name

        @property
        def value(self):
            return (self._value / self._n) if self._n > 0 else 0

    def __init__(self, params: Parameters, loss_only: bool = False):
        self._criterions = [
            Evaluation.Criterion('mse', torch.nn.MSELoss().cuda(), params.train.loss.mse),
            Evaluation.Criterion('ssim', SSIM(n_channels=params.data.sequence_length).cuda(), params.train.loss.ssim),
            Evaluation.Criterion('dice', lambda pred, gnd: tensor_zero(gnd.dtype), params.train.loss.dice),
            Evaluation.Criterion('loss', lambda pred, gnd: self._weighted_loss(pred, gnd)),
            Evaluation.Criterion('time', lambda pred, gnd: torch.tensor((datetime.now() - self._start).microseconds))
        ]
        self._getitem = {crit.name: c for c, crit in enumerate(self._criterions)}

        self._paths: dict[str, dict[str, float]] = {}
        self._loss_only = loss_only
        self._start = datetime.now()

    @property
    def paths(self) -> dict[str, dict[str, float]]:
        return self._paths

    @property
    def loss(self) -> torch.Tensor:
        return self._criterions[3].result

    def start_timer(self):
        if self._loss_only:
            raise AssertionError('only loss is available')
        self._start = datetime.now()

    def calculate(self, pred, gnd, path: str = None):
        """
        Calculate all criterions.
        """
        for crit in self._criterions:
            if self._loss_only and not crit.loss_weight and crit.name != 'loss':  # 0 or None
                continue

            if crit.name == 'dice':
                continue  # NYI
            elif crit.name == 'ssim':
                pred_permute, gnd_permute = pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0]
                crit.calculate(pred_permute, gnd_permute)
            else:
                crit.calculate(pred, gnd)

        if path:
            self._paths[path] = {crit.name: crit.result for crit in self._criterions}

    def __getitem__(self, item: str):
        if self._loss_only and item != 'loss':
            raise KeyError('only loss is available')
        value = self._criterions[self._getitem[item]].value
        if isinstance(value, torch.Tensor):
            return value.item()
        else:
            return value

    def _weighted_loss(self, pred, _) -> torch.Tensor:
        loss_sum = torch.tensor(0, device='cuda', dtype=pred.dtype)
        for crit in self._criterions:
            if crit.loss_weight:
                if crit.name == 'mse':
                    loss_sum += crit.loss_weight * crit.result
                else:
                    loss_sum += crit.loss_weight * (1 - crit.result)
        return loss_sum
