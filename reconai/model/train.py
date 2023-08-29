#!/usr/bin/env python
from __future__ import print_function, division

import datetime
import time

import torch
import torch.optim as optim
from typing import List
from pathlib import Path
import logging

from reconai.parameters import Parameters
from reconai.data.data import get_dataset_batchers, prepare_input_as_variable
from reconai.model.test import run_and_print_full_test

from reconai.utils.graph import update_loss_progress
from reconai.model.model_pytorch import CRNNMRI
from piqa import SSIM


def train(params: Parameters) -> List[tuple[int, List[int], List[int]]]:
    if not torch.cuda.is_available() and not params.debug:
        raise Exception('Can only run in Cuda')

    num_epochs = 3 if params.debug else params.config.train.epochs
    n_folds = params.config.train.folds if params.config.train.folds > 2 else 1
    undersampling = params.config.data.undersampling
    iterations = params.config.model.iterations

    # Configure directory info
    logging.info(f"saving model to {params.out_dir.absolute()}")

    # Specify network
    network = CRNNMRI(n_ch=params.config.model.channels,
                      nf=params.config.model.filters,
                      ks=params.config.model.kernelsize,
                      nc=2 if params.debug else iterations,
                      nd=params.config.model.layers,
                      single_crnn=params.config.data.equal_images and params.config.data.equal_masks
                      ).cuda()
    logging.info(f'# trainable parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}')
    optimizer = optim.Adam(network.parameters(), lr=float(params.config.train.lr), betas=(0.5, 0.999))
    if params.config.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.config.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.config.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.config.train.lr_gamma)

    train_val_batcher, test_batcher_equal, test_batcher_non_equal = get_dataset_batchers(params)

    results = []
    logging.info(f'started {n_folds}-fold training at {datetime.datetime.now()}')
    for fold in range(n_folds):
        fold_dir = params.out_dir / f'fold_{fold}'

        graph_train_err, graph_val_err = [], []
        mask_i = 0
        for epoch in range(num_epochs):
            logging.info(f'starting epoch {epoch}')
            t_start = time.time()

            network.train()
            train_err, train_batches = 0, 0
            for im in train_val_batcher.items_fold(fold, 5, validation=False):
                logging.debug(f"batch {train_batches}")
                im_u, k_u, mask, gnd = prepare_input_as_variable(im,
                                                                 params.config.train.mask_seed + mask_i,
                                                                 params.config.train.undersampling,
                                                                 params.config.train.equal_masks)

                optimizer.zero_grad(set_to_none=True)
                rec, full_iterations = network(im_u, k_u, mask)
                loss = calculate_loss(params, criterion, rec, gnd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1
                mask_i += params.config.data.sequence_length

                if params.debug and train_batches == 2:
                    break
            logging.info(f"completed {train_batches} train batches")

            validate_err, validate_batches = 0, 0
            network.eval()
            with torch.no_grad():
                for im in train_val_batcher.items_fold(fold, 5, validation=True):
                    logging.debug(f"batch {validate_batches}")
                    im_u, k_u, mask, gnd = prepare_input_as_variable(im,
                                                                     params.config.train.mask_seed + mask_i,
                                                                     params.config.train.undersampling,
                                                                     params.config.train.equal_masks)

                    pred, full_iterations = network(im_u, k_u, mask, test=True)
                    err = calculate_loss(params, criterion, pred, gnd)
                    validate_err += err.item()
                    validate_batches += 1

                    mask_i += params.config.data.sequence_length
                    if params.debug and validate_batches == 2:
                        break
            logging.info(f"completed {validate_batches} validate batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epochs}', f'\ttime: {t_end - t_start} s',
                               f'\ttraining loss:\t\t{train_err}x', f'\tvalidation loss:\t\t{validate_err}'])
            logging.info(stats)

            if epoch % 5 == 0 or epoch > num_epochs - 5:
                name = f'{params.name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), epoch_dir / npz_name)
                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')

                if epoch % 5 == 0 or epoch > num_epochs - 5:
                    run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, epoch_dir,
                                            name, train_err, validate_err, mask_i, stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            update_loss_progress(graph_train_err, graph_val_err, fold_dir, params.config.train.loss)
            log_epoch_stats_to_csv(fold_dir, undersampling, fold, epoch, train_err, validate_err)

            if params.config.train.stop_lr_decay == -1 or epoch < params.config.train.stop_lr_decay:
                scheduler.step()

        results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.datetime.now()}')

    return results


def get_criterion(params: Parameters) -> torch.nn.Module:
    if params.config.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.config.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.config.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    return criterion


def calculate_loss(params: Parameters, criterion, pred, gnd) -> float:
    if params.config.train.loss.mse == 1:
        err = criterion(pred, gnd)
    elif params.config.train.loss.ssim == 1:
        err = 1 - criterion(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])
    return err


def log_epoch_stats_to_csv(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float, val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')