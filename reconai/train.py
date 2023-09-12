#!/usr/bin/env python
from __future__ import print_function, division

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
import wandb
from piqa import SSIM

from reconai.data import SequenceBuilder, DataLoader, Batcher
from reconai.data.data import preprocess_as_variable
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import Parameters


def train(params: Parameters) -> List[tuple[int, List[int], List[int]]]:
    if not torch.cuda.is_available() and not params.debug:
        raise Exception('Can only run in Cuda')

    num_epochs = params.train.epochs
    n_folds = params.train.folds if params.train.folds > 2 else 1
    undersampling = params.data.undersampling
    mask_seed = params.data.mask_seed

    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    optimizer = optim.Adam(network.parameters(), lr=float(params.train.lr), betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.train.lr_gamma)
    # He initialization?
    if params.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    loader = DataLoader(params.in_dir).load(split_regex=params.data.split_regex, filter_regex=params.data.filter_regex)
    sequence_builder = SequenceBuilder(loader)
    sequence_collection = sequence_builder.generate_sequences(seed=params.data.sequence_seed,
                                                              seq_len=params.data.sequence_length)
    batcher = Batcher(loader)
    for sequence in sequence_collection.items():
        batcher.append_sequence(sequence, crop_expand_to=(params.data.shape_x, params.data.shape_y),
                                norm=params.data.normalize)

    logging.info(f"config.yaml: {str(params)}")
    logging.info(f"trainable parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
    logging.info(f"data: {len(loader)}, from {len(loader.keys())} cases")
    logging.info(f"saving model to {params.out_dir.absolute()}")
    params.mkoutdir()

    logging.info(f'starting {n_folds}-fold training')
    for fold in range(n_folds):
        fold_dir = params.out_dir / f'fold_{fold}'

        for epoch in range(num_epochs):
            epoch_start = datetime.now()

            network.train()
            train_loss, train_batches = 0, 0
            for im in batcher.items_fold(fold, n_folds, validation=False):
                im_u, k_u, mask, gnd = preprocess_as_variable(im, mask_seed, undersampling)

                optimizer.zero_grad(set_to_none=True)
                rec, full_iterations = network(im_u, k_u, mask)
                loss: torch.Tensor = calculate_loss(params, criterion, rec, gnd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                mask_seed += params.data.sequence_length

                if params.debug and train_batches == 2:
                    break
            network.eval()

            validate_loss, validate_batches = 0, 0
            with torch.no_grad():
                for im in batcher.items_fold(fold, 5, validation=True):
                    im_u, k_u, mask, gnd = preprocess_as_variable(im, mask_seed, undersampling)

                    pred, full_iterations = network(im_u, k_u, mask, test=True)
                    loss: torch.Tensor = calculate_loss(params, criterion, pred, gnd)

                    validate_loss += loss.item()
                    validate_batches += 1

                    mask_seed += params.data.sequence_length
                    if params.debug and validate_batches == 2:
                        break

            epoch_end = datetime.now()

            train_loss /= train_batches
            validate_loss /= validate_batches

            log = {'epoch': epoch,
                   'epoch_time': str(epoch_end - epoch_start),
                   'loss_train': train_loss,
                   'loss_eval': validate_loss, }

            logging.info('\n' + '\n'.join([f'{key}: {value:>15}' for key, value in log.items()]) + '\n')
            wandb.log(log)

            # rework this part

            # stats = '\n'.join([f'Epoch {epoch + 1}/{num_epochs}', f'\ttime: {t_end - t_start} s',
            #                    f'\ttraining loss:\t\t{train_loss}x', f'\tvalidation loss:\t\t{validate_loss}'])
            # logging.info(stats)
            #
            # if epoch % 5 == 0 or epoch > num_epochs - 5:
            #     name = f'{params.name}_fold_{fold}_epoch_{epoch}'
            #     npz_name = f'{name}.npz'
            #
            #     epoch_dir = fold_dir / name
            #     epoch_dir.mkdir(parents=True, exist_ok=True)
            #     torch.save(network.state_dict(), epoch_dir / npz_name)
            #     logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')
            #
            #     if epoch % 5 == 0 or epoch > num_epochs - 5:
            #         run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, epoch_dir,
            #                                 name, train_loss, validate_loss, mask_i, stats)
            #
            # graph_train_err.append(train_loss)
            # graph_val_err.append(validate_loss)
            #
            # update_loss_progress(graph_train_err, graph_val_err, fold_dir, params.train.loss)
            # log_epoch_stats_to_csv(fold_dir, undersampling, fold, epoch, train_loss, validate_loss)

            if params.train.lr_decay_end == -1 or epoch < params.train.lr_decay_end:
                scheduler.step()

        # results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.now()}')


def get_criterion(params: Parameters) -> torch.nn.Module:
    if params.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    return criterion


def calculate_loss(params: Parameters, criterion, pred, gnd) -> float:
    errors = []
    if params.train.loss.mse > 0:
        errors.append((params.train.loss.mse, criterion(pred, gnd)))
    if params.train.loss.ssim > 0:
        errors.append(
            (params.train.loss.mse, 1 - criterion(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])))
    if params.train.loss.dice > 0:
        raise NotImplementedError("Only MSE or SSIM loss implemented")
    return sum([w * err for w, err in errors])


def log_epoch_stats_to_csv(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float,
                           val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')
