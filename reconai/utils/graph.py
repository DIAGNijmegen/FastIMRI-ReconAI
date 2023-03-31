import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import math
from .metric import *


def print_loss_graphs():
    accelerations = [1, 2, 4, 8, 12, 16, 32, 64]

    graph_x = list(range(100))
    fig = plt.figure()
    for acceleration in accelerations:
        filename = f'../data/progress_{acceleration}.csv'
        frame = pd.read_csv(filename, delimiter=',')
        if acceleration == 64:
            graph_x = list(range(46))
        plt.plot(graph_x, frame.iloc[:, 3], label=f"train_loss_{acceleration}", lw=1)
    plt.legend()
    plt.ylim(bottom=0, top=0.008)
    plt.ylabel(f'mse training loss')
    plt.xlabel("epoch")
    plt.savefig('../data/training_loss.png')
    plt.close(fig)

    graph_x = list(range(100))
    fig = plt.figure()
    for acceleration in accelerations:
        filename = f'../data/progress_{acceleration}.csv'
        frame = pd.read_csv(filename, delimiter=',')
        if acceleration == 64:
            graph_x = list(range(46))
        plt.plot(graph_x, frame.iloc[:, 4], label=f"validation_loss_{acceleration}", lw=1)
    plt.legend()
    plt.ylim(bottom=0, top=0.005)
    plt.ylabel(f'mse validation loss')
    plt.xlabel("epoch")
    plt.savefig('../data/validation_loss.png')
    plt.close(fig)

def print_loss_progress(train_err, val_err, fold_dir: Path, loss: str):
    graph_x = list(range(len(train_err)))
    if len(graph_x) <= 1:
        return

    fig = plt.figure()
    plt.plot(graph_x, train_err, label="train_loss", lw=1)
    plt.plot(graph_x, val_err, label="val_loss", lw=1)
    plt.legend()
    plt.ylim(bottom=0)
    plt.ylabel(f'{loss} loss')
    plt.xlabel("epoch")
    plt.savefig(fold_dir / "progress.png")
    plt.close(fig)

def print_prediction_error(epoch_dir: Path, vis, name: str, validate_err: float):
    for i, (gnd, pred, und, seg) in enumerate(vis):
        fig = plt.figure()
        fig.suptitle(f'{name} (val loss: {validate_err})')
        axes = [plt.subplot(2, 3, j + 1) for j in range(3 * 2)]

        # gnd | pred | err
        axes, ax = set_ax(axes, 0, "ground truth", gnd[0])
        axes, ax = set_ax(axes, ax, "prediction 0", pred[0])
        axes, ax = set_ax(axes, ax, "error", gnd[0] - pred[0], cmap="magma")

        gnd_t, pred_t = gnd[..., gnd.shape[-1] // 2], pred[..., pred.shape[-1] // 2]
        axes, ax = set_ax(axes, ax, "ground truth", gnd_t)
        axes, ax = set_ax(axes, ax, "prediction", pred_t)
        axes, ax = set_ax(axes, ax, "error", gnd_t - pred_t, cmap="magma")

        fig.tight_layout()
        plt.savefig(epoch_dir / f'{name}.png', pad_inches=0)
        plt.close(fig)

def print_full_prediction_sequence(epoch_dir: Path, vis, name: str, validate_err: float, sequence_len: int, acceleration_factor: int):
    for i, (gnd, pred, und, seg) in enumerate(vis):
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(f'{name} (val loss: {validate_err})')
        axes = [plt.subplot(2, math.ceil(sequence_len / 2) + 1, j + 1) for j in range(sequence_len + 2)]

        axes, ax = set_ax(axes, 0, "ground truth", gnd[0])
        axes, ax = set_ax(axes, ax, f"{acceleration_factor}x undersampled", und[0])
        for k in range(sequence_len):
            axes, ax = set_ax(axes, ax, f"pred {k}", pred[k])

        fig.tight_layout()
        plt.savefig(epoch_dir / f'{name}_seq.png', pad_inches=0)
        plt.close(fig)

def print_loss_comparison_graphs(epoch_dir: Path, vis, name: str):
    for i, (gnd, pred, und, seg) in enumerate(vis):
        graph_x = list(range(len(gnd)))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        mse_val, psnr_val, ssim_val = [], [], []
        for j in graph_x:
            mse_val.append(mse(gnd[j], pred[j]))
            psnr_val.append(psnr(gnd[j], pred[j]))
            ssim_val.append(ssim(gnd[j], pred[j]))
        ax1.plot(graph_x, mse_val, label=f"mse", lw=1)
        ax1.set_title('MSE')
        ax2.plot(graph_x, psnr_val, label=f"psnr", lw=1)
        ax2.set_title('PSNR')
        ax3.plot(graph_x, ssim_val, label=f"ssim", lw=1)
        ax3.set_title('SSIM')
        plt.savefig(epoch_dir / f'{name}_errors.png')
        plt.close(fig)


def set_ax(axes, ax: int, title: str, image, cmap="Greys_r"):
    axes[ax].set_title(title)
    axes[ax].imshow(np.abs(image), cmap=cmap, interpolation="nearest", aspect='auto')
    axes[ax].set_axis_off()
    return axes, ax + 1
