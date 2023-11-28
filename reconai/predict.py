from pathlib import Path

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Prediction:
    def __init__(self, blob: np.ndarray, x: int, y: int, angle: float):
        self._blob = blob
        self._x = int(x)
        self._y = int(y)
        self._a = angle
        self._gnd: tuple | None = None

    @property
    def target(self) -> np.ndarray:
        return np.array([self._x, self._y], np.int32)

    @property
    def angle(self) -> float:
        return float(self._a)

    def set_ground_truth(self, x: int, y: int, angle: float):
        self._gnd = (x, y, angle)

    def show(self, save: Path = None):
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))

        axes.imshow(self._blob, cmap='gray')
        axes.set_axis_off()
        axes.set_title('Detected line')

        pred_gnd = [(self._x, self._y, self._a, 'b')] + ([(*self._gnd, 'g')] if self._gnd else [])
        for x, y, a, color in pred_gnd:
            x1, y1 = x + np.cos(a), y - np.sin(a)
            axes.axline((x1, y1), (x, y), color=color)
            axes.add_patch(plt.Circle((x, y), 3, color=color))
            axes.text(0, 10 if color == 'b' else 20, f'{np.rad2deg(a):.3f}', fontsize=12, color=color)

        plt.title(save.name[:50])
        if save:
            plt.savefig(save.as_posix())
        else:
            plt.show()


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


def predict_by_pca(blob: np.ndarray) -> Prediction | None:
    indices = np.array(np.where(blob > 0))
    center = np.mean(indices, axis=1, dtype=np.int32)
    # center seems incorrect?

    eigenvalues, eigenvectors = np.linalg.eig(np.cov(indices - center.reshape(2,1)))
    y, x = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate the angle of the major axis
    angle_radians = np.arctan2(-y, x)
    if -y < 0:
        angle_radians += np.pi

    return Prediction(blob, *walk_along_angle(blob, *center, np.arctan2(y, x)), angle=angle_radians)


def predict_by_hough_line_transform(blob: np.ndarray) -> Prediction | None:
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

    if target_pred:
        return Prediction(blob, *np.mean(target_pred, axis=0), angle=np.mean(angle_pred))
    else:
        return None


def predict_target(blob: np.ndarray, strategy: str = 'hough_line_transform') -> Prediction | None:
    bz, by, bx = blob.shape
    blob = blob[bz // 2, ...]
    if blob.max() == 0:
        return None

    match strategy:
        case 'hough_line_transform':
            return predict_by_hough_line_transform(blob)
        case 'pca':
            return predict_by_pca(blob)
        case _:
            raise ValueError(f'unknown strategy "{strategy}"')

