from pathlib import Path

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


angles = np.linspace(0, np.pi, 720, endpoint=False)


def hough_line_prediction(blob: np.ndarray, show: Path = None) -> tuple[np.ndarray | None, float]:
    bz, by, bx = blob.shape
    blob = blob[bz // 2, ...]
    blob_center = np.array([by // 2, bx // 2])

    if blob.max() == 0:
        return None, 0

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
        x0, y0 = peak_dist * np.array([np.cos(peak_angle), np.sin(peak_angle)])
        ax[2].axline((x0, y0), slope=np.tan(peak_angle + np.pi / 2))
        ax[2].add_patch(plt.Circle((x0, y0), 1, color='r'))
        for p in target_walk:
            ax[2].add_patch(plt.Circle((p[0], p[1]), 0.25, color='r'))
        if target_pred:
            ax[2].add_patch(plt.Circle(target_pred, 1.5, color='g'))
        ax[2].text(blob.shape[0] // 2, blob.shape[1] // 2, f'peak_angle: {np.rad2deg(peak_angle):.3f}\nangle_pred: {np.rad2deg(angle_pred):.3f}', fontsize=12, color='white', ha='right', va='bottom')

        plt.tight_layout()
        plt.show()
        plt.savefig(show.as_posix())

    return np.array(target_pred) if target_pred else None, -angle_pred
