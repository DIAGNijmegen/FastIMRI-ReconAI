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
        return float(self._a % (np.pi / 2))

    def set_ground_truth(self, x: int, y: int, angle: float):
        self._gnd = (x, y, angle)

    def show(self, save: Path = None):
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))

        axes.imshow(self._blob, cmap='gray')
        axes.set_axis_off()
        axes.set_title('Detected line')

        pred_gnd = [(self._x, self._y, self._a, 'b')] + ([(*self._gnd, 'g')] if self._gnd else [])
        for x, y, a, color in pred_gnd:
            x1, y1 = x + np.cos(a), y + np.sin(a)
            axes.axline((x1, y1), (x, y), color=color)
            axes.add_patch(plt.Circle((x, y), 3, color=color))
            axes.text(0, 10 if color == 'b' else 20, f'{np.rad2deg(a):.3f}', fontsize=12, color=color)

        plt.title(save.name)
        if save:
            plt.savefig(save.as_posix())
        else:
            plt.show()


def predict_by_hough_line_transform(blob: np.ndarray) -> Prediction | None:
    shape = np.array(blob.shape)
    h, theta, d = hough_line(blob, theta=np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False))

    target_pred = []
    angle_pred = []

    for peak in np.array(hough_line_peaks(h, theta, d)).transpose():
        # get hough point and a point perpendicular to it
        _, peak_angle, peak_dist = peak
        points = np.array([peak_angle, peak_angle + np.pi / 2])
        x0, x1, y0, y1 = peak_dist * np.array([np.cos(points), np.sin(points)]).flatten()
        x1, y1 = x0 + x1, y0 + y1

        # walk in both directions, see if we hit the segmentation
        walk_dir = np.arctan2(y1 - y0, x1 - x0)
        start = np.array([x0, y0])
        walk = [np.round(start + step * np.array([np.cos(walk_dir), np.sin(walk_dir)])) for step in range(-shape[0] * 2, shape[1] * 2)]
        hit = []
        for w in walk:
            x, y = int(w[0]), int(w[1])
            hit.append(bool(blob[y, x]) if 0 <= x < shape[0] and 0 <= y < shape[1] else False)

        # no hits, this hough line is bad
        if not np.any(hit):
            continue


        # heuristic: add the edge closest to the center of the image
        edges = [walk[h] for h in range(1, len(hit) - 1) if hit[h] and (not hit[h - 1] or not hit[h + 1])]
        target_pred.append(edges[np.argmin(np.linalg.norm(shape // 2 - edges, axis=1))])
        angle_pred.append(walk_dir)

        # fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        #
        # axes.imshow(blob, cmap='gray')
        # # axes.set_ylim((blob.shape[0], 0))
        # axes.set_axis_off()
        # axes.set_title('Detected lines')
        #
        # # axes.axline((x0, y0), (x1, y1), color='r')
        # axes.add_patch(plt.Circle((x0, y0), 3, color='r'))
        # axes.add_patch(plt.Circle((x1, y1), 3, color='b'))
        #
        # for w in walk:
        #     axes.add_patch(plt.Circle((w[0], w[1]), 1, color='g'))
        #
        # plt.show()
        # return
    if target_pred:
        return Prediction(blob, *np.mean(target_pred, axis=0), angle=np.mean(angle_pred))
    else:
        return None


def predict_target(blob: np.ndarray) -> Prediction | None:
    bz, by, bx = blob.shape
    blob = blob[bz // 2, ...]
    if blob.max() == 0:
        return None

    return predict_by_hough_line_transform(blob)


    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    axes.imshow(blob, cmap='gray')
    # axes.set_ylim((blob.shape[0], 0))
    axes.set_axis_off()
    axes.set_title('Detected lines')

    axes.add_patch(plt.Circle(target, 3, color='r'))
    x, y = np.array(target) + [np.cos(angle), np.sin(angle)]
    axes.axline((x, y), target, color='r')

    plt.savefig(show.as_posix())

    return target, angle

    h, theta, d = hough_line(blob, theta=angles)
    peaks = np.array(hough_line_peaks(h, theta, d)).transpose()

    _, peak_angle, peak_dist = peaks[0]
    x0, y0 = peak_dist * np.array([np.cos(peak_angle), np.sin(peak_angle)])
    angle_pred = np.tan(peak_angle) + np.pi / 2

    target_walk, target_hit = [], []
    for dist in range(-len(blob), len(blob)):
        x = int(np.round(x0 + dist * np.cos(angle_pred)))
        y = int(np.round(y0 + dist * np.sin(angle_pred)))
        if 0 <= x < bx and 0 <= y < by:
            if (x, y) not in target_walk:
                target_walk.append((x, y))
                target_hit.append(bool(blob[y, x]))

    target_pred: tuple | None = None
    target_dist: float = np.inf
    for i, (x, y) in enumerate(target_walk):
        if 0 < i < len(target_walk) - 1:
            if target_hit[i] and (not target_hit[i - 1] or not target_hit[i + 1]):
                # it's an edge
                dist_pred = np.linalg.norm(blob_center - np.array([y, x]))
                if dist_pred < target_dist:
                    target_pred = (x, y)
                    target_dist = dist_pred
    # issue on the third instance!

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(blob, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                  np.rad2deg(theta[-1] + angle_step),
                  d[-1] + d_step, d[0] - d_step]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap='gray', aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(blob, cmap='gray')
        ax[2].set_ylim((blob.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        # ax[2].axline(blob_center, blob_center + (np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40))), color='white')
        # ax[2].add_patch(plt.Circle(blob_center + (np.cos(np.deg2rad(40)) * 10, np.sin(np.deg2rad(40)) * 10), 2, color='white'))
        # for s in np.linspace(0,180,6, endpoint=False):
        #     ax[2].axline(blob_center, blob_center + (np.cos(np.deg2rad(s)), np.sin(np.deg2rad(s))), color='c')
        for p, peak in enumerate(peaks):
            _, peak_angle, peak_dist = peak
            x0, y0 = peak_dist * np.array([np.cos(peak_angle), np.sin(peak_angle)])
            x1, y1 = peak_dist * np.array([np.cos(peak_angle + np.pi / 2), np.sin(peak_angle + np.pi / 2)])
            x1, y1 = x0 + x1, y0 + y1
            ax[2].axline((x0, y0), (x1, y1), color=colors[p])
            ax[2].add_patch(plt.Circle((x0, y0), 3, color=colors[p]))
            ax[2].add_patch(plt.Circle((x1, y1), 2, color=colors[p]))
            ax_text = f'{p}: {np.rad2deg(peak_angle):.3f}, x0, y0: ({x0:.2f}, {y0:.2f}), x1, y1: ({x1:.2f}, {y1:.2f})'
            ax[2].text(blob.shape[0] // 2, blob.shape[1] // 2 - p * 10, ax_text, fontsize=12, color=colors[p])

        # for p in target_walk:
        #     ax[2].add_patch(plt.Circle((p[0], p[1]), 0.25, color='r'))
        # if target_pred:
        #     ax[2].add_patch(plt.Circle(target_pred, 1.5, color='g'))
        # ax[2].text(blob.shape[0] // 2, blob.shape[1] // 2, f'peak_angle: {np.rad2deg(peak_angle):.3f}\nangle_pred: {np.rad2deg(angle_pred):.3f}', fontsize=12, color='white', ha='right', va='bottom')

        plt.tight_layout()
        # plt.show()
        plt.savefig(show.as_posix())

    return np.array(target_pred) if target_pred else None, -angle_pred
