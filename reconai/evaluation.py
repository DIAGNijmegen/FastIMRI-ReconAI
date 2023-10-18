from datetime import datetime
from typing import Callable

import numpy as np
import torch
from piqa import SSIM

from .parameters import Parameters


def tensor_zero(dtype: torch.dtype):
    return torch.tensor(0, device='cuda', dtype=dtype)


class Evaluation:
    class Criterion:
        def __init__(self, name: str, criterion: Callable, loss_weight: float | None = None):
            self._name = name
            self._loss_weight = loss_weight
            self._criterion = criterion
            self._n = -1
            self._value = None
            self._result = None
            self._min_max = None

        def __str__(self):
            return f'{self.name} ({self._loss_weight})'

        def calculate(self, pred: torch.Tensor, gnd: torch.Tensor):
            self._result: torch.Tensor = self._criterion(pred, gnd)
            result = self._result.item()

            if not self._value:
                if self.name == 'time' and self._n < 0:
                    self._n = 0
                    self._result = None
                    return  # the very first inference includes start up time we wish to ignore

                self._n = 0
                self._value = tensor_zero(self._result.dtype)
                self._min_max = [result, result]

            if result < self._min_max[0]:
                self._min_max[0] = result
            if result > self._min_max[1]:
                self._min_max[1] = result

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

        @property
        def min(self) -> float:
            if not self._value:
                return 0
            return self._min_max[0]

        @property
        def max(self) -> float:
            if not self._value:
                return 0
            return self._min_max[1]

    def __init__(self, params: Parameters, loss_only: bool = False):
        self._criterions = [
            Evaluation.Criterion('mse', torch.nn.MSELoss().cuda(), params.train.loss.mse),
            Evaluation.Criterion('ssim', SSIM(n_channels=params.data.sequence_length).cuda(), params.train.loss.ssim),
            Evaluation.Criterion('dice', self._dice, params.train.loss.dice),
            Evaluation.Criterion('loss', self._weighted_loss),
            Evaluation.Criterion('time', self._time)
        ]
        self._getitem = {crit.name: c for c, crit in enumerate(self._criterions)}

        self._keys: dict[str, dict[str, float]] = {}
        self._loss_only = loss_only
        self._start: datetime | None = None

    @property
    def loss(self) -> torch.Tensor:
        return self._criterions[self._getitem['loss']].result

    @property
    def criterion_value_per_key(self) -> dict[str, dict[str, float]]:
        return self._keys

    def criterion_stats(self, name: str) -> tuple[float, float, float]:
        if self._loss_only and name != 'loss':
            raise NameError('only loss is available')
        crit = self._criterions[self._getitem[name]]
        return crit.min, crit.value, crit.max

    def start_timer(self):
        if self._loss_only:
            raise AssertionError('only loss is available')
        self._start = datetime.now()

    def calculate_dice(self, pred, gnd, key: str = None):
        crit_dice = self._criterions[self._getitem['dice']]
        crit_dice.calculate(pred, gnd)
        if key:
            self._keys[key] = self._keys.get(key, {}) | {'dice': crit_dice.result.item()}

    def calculate(self, pred: torch.Tensor, gnd: torch.Tensor, key: str = None):
        """
        Calculate all criterions.
        """
        pred, gnd = torch.nan_to_num(pred, nan=0.0), torch.nan_to_num(gnd, nan=0.0)
        for crit in self._criterions:
            if self._loss_only and not crit.loss_weight and crit.name != 'loss':  # 0 or None
                continue

            if crit.name == 'dice':
                continue
            elif crit.name == 'time' and self._start is None:
                continue
            elif crit.name == 'ssim':
                crit.calculate(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])
            else:
                crit.calculate(pred, gnd)

        if key:
            stats = {crit.name: crit.result.item() for crit in self._criterions if crit.result is not None}
            self._keys[key] = self._keys.get(key, {}) | stats

    def _weighted_loss(self, pred, _) -> torch.Tensor:
        loss_sum = torch.tensor(0, device='cuda', dtype=pred.dtype)
        for crit in self._criterions:
            if crit.loss_weight:
                if crit.name == 'mse':
                    loss_sum += crit.loss_weight * crit.result
                else:
                    loss_sum += crit.loss_weight * (1 - crit.result)
        return loss_sum

    def _time(self, _, __) -> torch.Tensor:
        dt = datetime.now() - self._start
        ms = (dt.seconds * 1e3 + dt.microseconds / 1e3)
        return torch.tensor(ms)

    @staticmethod
    def _dice(pred, gnd) -> torch.Tensor:
        intersection = np.sum(pred[gnd == 1]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(gnd))
        return torch.tensor(dice)
