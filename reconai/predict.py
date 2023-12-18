from pathlib import Path

import matplotlib
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import SimpleITK as sitk

import matplotlib.pyplot as plt

prediction_strategies = ['pca', 'hough_line_transform']


class Prediction:
    def __init__(self, blob: np.ndarray, gnd: tuple[int, int, float], pred: tuple[int, int, float]):
        self._blob = blob.squeeze()
        self._gnd_target = np.array(gnd[:2], np.int32)
        self._gnd_angle = np.array(gnd[-1], np.int32)
        self._pred_target = np.array(pred[:2], np.int32)
        self._pred_angle = np.array(pred[-1], np.int32)
        self._failed = sum(pred) == 0

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

