from typing import Callable
from datetime import datetime
from pathlib import Path

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
            self._value = None
            self._result = None

        def __str__(self):
            return f'{self.name} ({self._loss_weight})'

        def calculate(self, pred: np.ndarray, gnd: np.ndarray):
            if not self._value:
                self._value = tensor_zero(gnd.dtype)

            self._result: torch.Tensor = self._criterion(pred, gnd)
            self._n += 1
            self._value += self._result

        @property
        def result(self) -> torch.Tensor | None:
            return self._result

        @property
        def loss_weight(self) -> float | None:
            return self._loss_weight

        @property
        def name(self) -> str:
            return self._name

        @property
        def value(self) -> float:
            if not self._value:
                return 0
            return (self._value / self._n).item()

    def __init__(self, params: Parameters, loss_only: bool = False):
        self._criterions = [
            Evaluation.Criterion('mse', torch.nn.MSELoss().cuda(), params.train.loss.mse),
            Evaluation.Criterion('ssim', SSIM(n_channels=params.data.sequence_length).cuda(), params.train.loss.ssim),
            Evaluation.Criterion('dice', lambda pred, gnd: tensor_zero(gnd.dtype), params.train.loss.dice),
            Evaluation.Criterion('loss', lambda pred, gnd: self._weighted_loss(pred, gnd)),
            Evaluation.Criterion('time', lambda pred, gnd: torch.tensor((datetime.now() - self._start).microseconds))
        ]
        self._getitem = {crit.name: c for c, crit in enumerate(self._criterions)}

        self._keys: dict[str, dict[str, float]] = {}
        self._loss_only = loss_only
        self._start: datetime | None = None

    @property
    def loss(self) -> torch.Tensor:
        return self._criterions[3].result

    @property
    def criterion_value_per_key(self) -> dict[str, dict[str, float]]:
        return self._keys

    def criterion_value(self, name: str) -> float:
        if self._loss_only and name != 'loss':
            raise NameError('only loss is available')
        return self._criterions[self._getitem[name]].value

    def start_timer(self):
        if self._loss_only:
            raise AssertionError('only loss is available')
        self._start = datetime.now()

    def calculate(self, pred, gnd, key: str = None):
        """
        Calculate all criterions.
        """
        for crit in self._criterions:
            if self._loss_only and not crit.loss_weight and crit.name != 'loss':  # 0 or None
                continue

            if crit.name == 'time' and self._start is None:
                continue
            elif crit.name == 'dice':
                continue  # NYI
            elif crit.name == 'ssim':
                crit.calculate(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])
            else:
                crit.calculate(pred, gnd)

        if key:
            self._keys[key] = {crit.name: crit.value for crit in self._criterions if crit.result is not None}

    def _weighted_loss(self, pred, _) -> torch.Tensor:
        loss_sum = torch.tensor(0, device='cuda', dtype=pred.dtype)
        for crit in self._criterions:
            if crit.loss_weight:
                if crit.name == 'mse':
                    loss_sum += crit.loss_weight * crit.result
                else:
                    loss_sum += crit.loss_weight * (1 - crit.result)
        return loss_sum
