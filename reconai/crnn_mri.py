#!/usr/bin/env python
from __future__ import print_function, division

import ctypes
import datetime
import random
import time
import logging
from pathlib import Path
import math

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from box import Box

from reconai.utils.metric import complex_psnr
from reconai.cascadenet_pytorch.module import Module
from reconai.cascadenet_pytorch.model_pytorch import CRNN_MRI
from reconai.cascadenet_pytorch.dnn_io import from_tensor_format
from .utils.kspace import kspace_to_image, image_to_kspace
from .data.data import get_data_volumes, get_dataset_batchers, prepare_input

from .data.Volume import Volume

def iimshow(im):
    plt.imshow(np.abs(im.squeeze()), cmap="Greys_r")
    plt.show()

def test_accelerations(args: Box):
    accelerations = [1, 2, 4, 8, 12, 16, 32]

    results = []
    for acceleration in accelerations:
        args['acceleration_factor'] = acceleration
        train_results = train_network(args, True)
        fold0, train_err, val_err = train_results[0]
        results.append((acceleration, train_err, val_err))

    graph_x = list(range(args.num_epoch))
    fig = plt.figure()
    for acceleration, train_err, val_err in results:
        plt.plot(graph_x, train_err, label=f"train_loss_{acceleration}", lw=1)
    plt.legend()
    plt.ylim(bottom=0)
    plt.ylabel(f'{args.loss} training loss')
    plt.xlabel("epoch")
    plt.savefig(args.out_dir / f'acceleration_{args.date}' / "training_loss.png")
    plt.close(fig)

    fig = plt.figure()
    for acceleration, train_err, val_err in results:
        plt.plot(graph_x, val_err, label=f"val_loss{acceleration}", lw=1)
    plt.legend()
    plt.ylim(bottom=0)
    plt.ylabel(f'{args.loss} validation loss')
    plt.xlabel("epoch")
    plt.savefig(args.out_dir / f'acceleration_{args.date}' / "validation_loss.png")
    plt.close(fig)


def train_network(args: Box, test_acc: bool = False) -> ctypes.Array:
    # change relu to leakyrelu, change loss to 0-1 somehow
    # Project config
    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    # normalization_test(args)
    # exit(1)
    # Volume.key = 'needle'
    model_name = f'crnn_mri_debug' if args.debug else f'crnn_mri'
    num_epoch = 3 if args.debug else args.num_epoch
    n_folds = args.folds if args.folds > 2 else 1
    seg_ai_dir = args.seg_ai_dir

    # Configure directory info
    save_dir: Path = \
        args.out_dir / f'acceleration_{args.date}' / f'{model_name}_acceleration_{args.acceleration_factor}' \
        if test_acc else args.out_dir / f'{model_name}_{args.date}'
    save_dir.mkdir(parents=True)
    logging.info(f"saving model to {save_dir.absolute()}")

    # Specify network
    rec_net = CRNN_MRI(n_ch=1, nc=2 if args.debug else 5).cuda()
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr), betas=(0.5, 0.999))
    if args.loss == 'ssim':
        segAI = True
        raise NotImplementedError()
    elif args.loss == 'mse+ssim':
        segAI = True
        raise NotImplementedError()
    else:  # 'mse'
        segAI = False
        criterion = torch.nn.MSELoss().cuda()

    if segAI and not seg_ai_dir:
        raise ValueError(f'missing {seg_ai_dir} directory')

    data = get_data_volumes(args)

    results = []
    logging.info(f'started {n_folds}-fold training at {datetime.datetime.now()}')
    for fold in range(n_folds):
        fold_dir = save_dir / f'fold_{fold}'

        graph_train_err, graph_val_err = [], []
        for epoch in range(num_epoch):
            t_start = time.time()

            train, validate, test = get_dataset_batchers(args, data, n_folds, fold)
            train_err, train_batches = 0, 0
            for im in train.generate():
                logging.debug(f"batch {train_batches}")
                im_und, k_und, mask, im_gnd = prepare_input(im, args.acceleration_factor)
                im_u = Variable(im_und.type(Module.TensorType))
                k_u = Variable(k_und.type(Module.TensorType))
                mask = Variable(mask.type(Module.TensorType))
                gnd = Variable(im_gnd.type(Module.TensorType))

                optimizer.zero_grad()
                rec = rec_net(im_u, k_u, mask)
                loss = criterion(rec, gnd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rec_net.parameters(), max_norm=5)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1

                if args.debug and train_batches == 2:
                    break
            logging.info(f"completed {train_batches} train batches")

            validate_err, validate_batches = 0, 0
            rec_net.eval()
            with torch.no_grad():
                for im in validate.generate():
                    logging.debug(f"batch {validate_batches}")
                    im_und, k_und, mask, im_gnd = prepare_input(im, args.acceleration_factor)
                    im_u = Variable(im_und.type(Module.TensorType))
                    k_u = Variable(k_und.type(Module.TensorType))
                    mask = Variable(mask.type(Module.TensorType))
                    gnd = Variable(im_gnd.type(Module.TensorType))

                    pred = rec_net(im_u, k_u, mask, test=True)
                    err = criterion(pred, gnd)

                    validate_err += err.item()
                    validate_batches += 1

                    if args.debug and validate_batches == 2:
                        break
            logging.info(f"completed {validate_batches} validate batches")

            vis = []
            base_psnr = 0
            test_psnr = 0
            test_batches = 0
            with torch.no_grad():
                for im in test.generate():
                    logging.debug(f"batch {test_batches}")
                    # scaler = MinMaxScaler()
                    # for layer in range(im.shape[1]):
                    #     im[0, layer, :, :] = scaler.fit_transform(im[0, layer, :, :])
                    im_und, k_und, mask, im_gnd = prepare_input(im, args.acceleration_factor)
                    im_u = Variable(im_und.type(Module.TensorType))
                    k_u = Variable(k_und.type(Module.TensorType))
                    mask = Variable(mask.type(Module.TensorType))

                    pred = rec_net(im_u, k_u, mask, test=True)

                    for im_i, und_i, pred_i in zip(im,
                                                   from_tensor_format(im_und.numpy()),
                                                   from_tensor_format(pred.data.cpu().numpy())):
                        base_psnr += complex_psnr(im_i, und_i, peak='max')
                        test_psnr += complex_psnr(im_i, pred_i, peak='max')

                    if test_batches == 0:  # save n samples
                        vis.append((from_tensor_format(im_gnd.numpy())[0],
                                    from_tensor_format(pred.data.cpu().numpy())[0],
                                    0 if not segAI else 0))

                    test_batches += 1
                    if args.debug and test_batches == 2:
                        break
            logging.info(f"completed {test_batches} test batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches
            base_psnr /= (test_batches * args.batch_size)
            test_psnr /= (test_batches * args.batch_size)

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epoch}',
                     f'\ttime: {t_end - t_start} s',
                     f'\ttraining loss:\t\t{train_err}',
                     f'\tvalidation loss:\t\t{validate_err}',
                     f'\tbase PSNR:\t\t\t{base_psnr}',
                     f'\ttest PSNR:\t\t\t{test_psnr}'])
            logging.info(stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            # graph_train_err_norm = [err / np.linalg.norm(graph_train_err, ord=np.inf) for err in graph_train_err]
            # graph_val_err_norm = [err / np.linalg.norm(graph_val_err, ord=np.inf) for err in graph_val_err]

            # TODO: put this in separate functions
            if epoch > 0:
                graph_x = list(range(len(graph_train_err)))
                fig = plt.figure()
                plt.plot(graph_x, graph_train_err, label="train_loss", lw=1)
                plt.plot(graph_x, graph_val_err, label="val_loss", lw=1)
                plt.legend()
                plt.ylim(bottom=0)
                plt.ylabel(f'{args.loss} loss')
                plt.xlabel("epoch")
                plt.savefig(fold_dir / "progress.png")
                plt.close(fig)

            if epoch % 5 == 0 or epoch > num_epoch - 5:
                name = f'{model_name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)
                for i, (gnd, pred, seg) in enumerate(vis):
                    fig = plt.figure()
                    fig.suptitle(f'{name} (val loss: {validate_err})')
                    axes = [plt.subplot(2, 4 if segAI else 3, j + 1) for j in range(4 if segAI else 3 * 2)]
                    ax: int = 0

                    def set_ax(title: str, im, cmap="Greys_r"):
                        nonlocal ax
                        axes[ax].set_title(title)
                        axes[ax].imshow(np.abs(im), cmap=cmap, interpolation="nearest", aspect='auto')
                        axes[ax].set_axis_off()
                        ax = ax + 1

                    # gnd | pred | err | seg
                    set_ax("ground truth", gnd[0])
                    set_ax("prediction 0", pred[0])
                    set_ax("error", gnd[0] - pred[0], cmap="magma")

                    gnd_t, pred_t = gnd[..., gnd.shape[-1] // 2], pred[..., pred.shape[-1] // 2]
                    set_ax("ground truth", gnd_t)
                    set_ax("prediction", pred_t)
                    set_ax("error", gnd_t - pred_t, cmap="magma")

                    fig.tight_layout()
                    plt.savefig(epoch_dir / f'{name}.png', pad_inches=0)
                    plt.close(fig)

                for i, (gnd, pred, seg) in enumerate(vis):
                    fig = plt.figure(figsize=(20, 8))
                    fig.suptitle(f'{name} (val loss: {validate_err})')
                    axes = [plt.subplot(2, 4 if segAI else math.ceil(args.sequence_len / 2), j+1) for j in range(args.sequence_len + 1)]
                    ax: int = 0

                    def set_ax(title: str, im, cmap="Greys_r"):
                        nonlocal ax
                        axes[ax].set_title(title)
                        axes[ax].imshow(np.abs(im), cmap=cmap, interpolation="nearest", aspect='auto')
                        axes[ax].set_axis_off()
                        ax = ax + 1

                    set_ax("ground truth", gnd[0])
                    for k in range(args.sequence_len):
                        set_ax(f"prediction {k}", pred[k])

                    fig.tight_layout()
                    plt.savefig(epoch_dir / f'{name}_seq.png', pad_inches=0)
                    plt.close(fig)

                torch.save(rec_net.state_dict(), epoch_dir / npz_name)
                with open(epoch_dir / f'{name}.log', 'w') as log:
                    log.write(stats)

                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')
            append_to_file(fold_dir, args.acceleration_factor, fold, epoch, train_err, validate_err)
        results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.datetime.now()}')

    return results

def append_to_file(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float, val_err: float):
    with open(fold_dir / f'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write(f'Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')


def normalization_test(args):
    def show_im(original, minmaxscaled, standardscaled, sampledimageminmax, sampledimagestandard):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(original)
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(minmaxscaled)
        axs[0, 1].set_title('MinMaxScaled')
        axs[0, 2].imshow(sampledimageminmax)
        axs[0, 2].set_title('Sampled')
        axs[0, 3].imshow(sampledimageminmax - minmaxscaled)
        axs[0, 3].set_title('MM - S diff')
        axs[1, 0].imshow(original)
        axs[1, 0].set_title('Original Image')
        axs[1, 1].imshow(standardscaled)
        axs[1, 1].set_title('StandardScaled')
        axs[1, 2].imshow(sampledimagestandard)
        axs[1, 2].set_title('Sampled')
        axs[1, 3].imshow(sampledimagestandard - standardscaled)
        axs[1, 3].set_title('SS - S diff')
        plt.show()

    Volume.key = 'needle'
    data = get_data_volumes(args)
    train, _, _ = get_dataset_batchers(args, data, 1, 0)

    image = next(train.generate())

    scaler = MinMaxScaler()
    minmax_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        minmax_scaled_image[0, layer, :, :] = scaler.fit_transform(minmax_scaled_image[0, layer, :, :])

    scaler = StandardScaler()
    standard_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        standard_scaled_image[0, layer, :, :] = scaler.fit_transform(standard_scaled_image[0, layer, :, :])

    sampled_mm = kspace_to_image(image_to_kspace(minmax_scaled_image[0][0]))
    sampled_ss = kspace_to_image(image_to_kspace(standard_scaled_image[0][0]))

    show_im(image[0][0], minmax_scaled_image[0][0], standard_scaled_image[0][0], sampled_mm, sampled_ss)

    exit(1)

def get_data_information(args):
    # Volume.key = 'needle'
    data = get_data_volumes(args)
    train, validate, _ = get_dataset_batchers(args, data, 1, 0)


    mins = []
    maxes = []
    max_to_perc99 = []
    averages = []
    for image in train.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            mins.append(slice.min())
            maxes.append(slice.max())
            averages.append(slice.mean())
            perc99 = np.percentile(slice, 99)
            max_to_perc99.append(slice.max() - perc99)

    for image in validate.generate():
        image = image[0]

        # go through each slice
        for i in range(image.shape[0]):
            slice = image[i]
            maxes.append(slice.max())

    # print(f'Min {min(mins)}')
    print(f'Min of maxes {min(maxes)}')
    print(f'Average of maxes {np.mean(maxes)}')
    print(f'std of maxes {np.std(maxes)}')
    print(f'Max of maxes {max(maxes)}')
    # print(f'Avg {np.mean(averages)}')
    # print(f'Min diff max-perc99 {np.min(max_to_perc99)}')
    # print(f'Avg diff max-perc99 {np.mean(max_to_perc99)}')
    # print(f'Max diff max-perc99 {np.max(max_to_perc99)}')
    exit(1)






