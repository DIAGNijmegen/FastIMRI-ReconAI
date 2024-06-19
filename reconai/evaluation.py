from datetime import datetime
from pathlib import Path
from typing import Callable

import SimpleITK as sitk
import numpy as np
import torch
from piqa import SSIM
from skimage.transform import hough_line, hough_line_peaks

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
            Evaluation.Criterion('time', self._time),
            Evaluation.Criterion('target', self._target_direction),
            Evaluation.Criterion('direction', self._target_direction)
        ]
        self._getitem = {crit.name: c for c, crit in enumerate(self._criterions)}

        self._results: dict[str, dict[str, float]] = {}
        self._loss_only = loss_only
        self._start: datetime | None = None

    @property
    def loss(self) -> torch.Tensor:
        return self._criterions[self._getitem['loss']].result

    @property
    def criterion_value_per_key(self) -> dict[str, dict[str, float]]:
        return self._results

    def _add_dict_to_results(self, key: str, dictionary: dict):
        if key:
            self._results[key] = self._results.get(key, {}) | dictionary

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
            self._add_dict_to_results(key, {'dice': crit_dice.result.item()})

    def calculate_target_direction(self, pred, gnd, spacing: tuple[float, float] = (1, 1), strategy: str = None, key: str = None):
        prediction = predict(pred, gnd, strategy=strategy)
        target, direction = prediction.error(spacing=spacing)
        self._add_dict_to_results(key, {'target_gnd': prediction.gnd_target, 'direction_gnd': prediction.gnd_target})
        for item, pred, error in zip(['target', 'direction'],
                                     [prediction.pred_target, prediction.pred_angle],
                                     [target, direction]):
            crit = self._criterions[self._getitem[item]]
            crit.calculate(error, torch.tensor(0))
            if key:
                stats = {f'{item}_{strategy}': crit.result.item(), f'{item}_pred_{strategy}': pred}
                self._add_dict_to_results(key, stats)

    def calculate_reconstruction(self, pred: torch.Tensor, gnd: torch.Tensor, key: str = None):
        """
        Calculate all criterions.
        """
        pred, gnd = torch.nan_to_num(pred, nan=0.0), torch.nan_to_num(gnd, nan=0.0)
        for crit in self._criterions:
            if self._loss_only and not crit.loss_weight and crit.name != 'loss':  # 0 or None
                continue

            if crit.name in ['dice', 'target', 'direction']:
                continue
            elif crit.name == 'time' and self._start is None:
                continue
            elif crit.name == 'ssim':
                crit.calculate(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])
            else:
                crit.calculate(pred, gnd)

        if key:
            stats = {crit.name: crit.result.item() for crit in self._criterions if crit.result is not None}
            self._add_dict_to_results(key, stats)

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
        return torch.tensor(0) if np.isnan(dice) else torch.tensor(dice)

    @staticmethod
    def _target_direction(pred, _) -> torch.Tensor:
        return torch.tensor(pred, dtype=torch.float32)


class Prediction:
    STRATEGIES = ['pca', 'hough_line_transform']

    def __init__(self, blob: np.ndarray, gnd: tuple[int, int, float], pred: tuple[int, int, float]):
        self._blob = blob.squeeze()
        self._gnd_target = np.array(gnd[:2], np.int32)
        self._gnd_angle = np.array(gnd[-1], np.float32)
        self._pred_target = np.array(pred[:2], np.int32)
        self._pred_angle = np.array(pred[-1], np.float32)
        self._failed = sum(pred) == 0

    @property
    def gnd_angle(self) -> float:
        return -1 if self._failed else float(self._gnd_angle)

    @property
    def gnd_target(self) -> tuple[int, int]:
        return (-1, -1) if self._failed else tuple(self._gnd_target.tolist())

    @property
    def pred_angle(self) -> float:
        return -1 if self._failed else float(self._pred_angle)

    @property
    def pred_target(self) -> tuple[int, int]:
        return (-1, -1) if self._failed else tuple(self._pred_target.tolist())

    @property
    def failed(self) -> bool:
        return self._failed

    def error(self, spacing: tuple[float, float] = (1, 1)) -> tuple[float, float]:
        if self._failed:
            return -1, -1

        target_error = np.linalg.norm((np.multiply(self._gnd_target - self._pred_target, spacing)))
        angle_error = np.rad2deg(np.abs(self._gnd_angle - self._pred_angle))
        return float(target_error), float(angle_error)

    def save(self, file: Path, debug: bool = False):
        if debug:
            blob = (self._blob * 255).astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(blob), file.parent / f'{file.stem}_blob{file.suffix}')

        for target, name in [(self._gnd_target, 'gnd'), (self._pred_target, 'pred')]:
            x, y = target
            arr = np.zeros_like(self._blob, dtype=np.uint8)
            if name == 'gnd' or not self._failed:
                arr[y, x] = 255
            sitk.WriteImage(sitk.GetImageFromArray(arr), file.parent / f'{file.stem}_{name}{file.suffix}')


def walk_along_angle(blob: np.ndarray, start_x: int, start_y: int, direction: float) -> np.ndarray:
    shape = np.array(blob.shape)
    start = np.array([start_x, start_y])
    walk = [np.round(start + step * np.array([np.cos(direction), np.sin(direction)])) for step in
            range(-shape[0] * 2, shape[1] * 2)]

    hit = []
    for w in walk:
        x, y = int(w[0]), int(w[1])
        hit.append(bool(blob[y, x]) if 0 <= x < shape[0] and 0 <= y < shape[1] else False)

    if not np.any(hit):
        return np.array([start_x, start_y])

    # heuristic: add the edge closest to the center of the image
    edges = [walk[h] for h in range(1, len(hit) - 1) if hit[h] and (not hit[h - 1] or not hit[h + 1])]
    return edges[np.argmin(np.linalg.norm(shape // 2 - edges, axis=1))]


def predict_by_pca(blob: np.ndarray) -> tuple[int, int, float] | None:
    indices = np.array(np.where(blob > 0))
    center = np.flip(np.mean(indices, axis=1, dtype=np.int32))

    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(indices - center.reshape(2, 1)))
    except np.linalg.LinAlgError:
        return None
    y, x = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate the angle of the major axis
    angle_radians: float = np.arctan2(-y, x)
    if -y < 0:
        angle_radians += np.pi

    target_pred: list[int] = list(walk_along_angle(blob, *center, np.arctan2(y, x)))
    return target_pred[0], target_pred[1], angle_radians


def predict_by_hough_line_transform(blob: np.ndarray) -> tuple[int, int, float] | None:
    h, theta, d = hough_line(blob, theta=np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False))

    target_pred = []
    angle_pred = []

    for peak in np.array(hough_line_peaks(h, theta, d)).transpose():
        # get hough point and a point perpendicular to it
        _, peak_angle, peak_dist = peak
        points = np.array([peak_angle, peak_angle + np.pi / 2])
        x0, x1, y0, y1 = peak_dist * np.array([np.cos(points), np.sin(points)]).flatten()
        x1, y1 = x0 + x1, y0 + y1
        dy, dx = y1 - y0, x1 - x0

        target_pred.append(walk_along_angle(blob, x0, y0, np.arctan2(dy, dx)))

        angle = np.arctan2(-dy, dx)
        if -dy < 0:
            angle += np.pi
        angle_pred.append(angle % (np.pi * 2))

    return (*np.mean(target_pred, axis=0), np.mean(angle_pred)) if target_pred else None


def predict(blob: np.ndarray, gnd: tuple[int, int, float], strategy: str = 'pca') -> Prediction:
    blob = blob.squeeze()
    assert len(blob.shape) == 2, 'blob not a 2-dimensional array'

    no_prediction = Prediction(blob, gnd, (0, 0, 0))
    if blob.max() == 0:
        return no_prediction

    match strategy:
        case 'hough_line_transform':
            prediction = predict_by_hough_line_transform(blob)
        case 'pca' | None:
            prediction = predict_by_pca(blob)
        case _:
            raise ValueError(f'unknown strategy "{strategy}"')

    return Prediction(blob, gnd, prediction) if prediction else no_prediction
