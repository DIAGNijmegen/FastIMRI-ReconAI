#!/usr/bin/env python
from __future__ import print_function, division

import datetime
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from box import Box
from typing import List
from pathlib import Path
import logging

from reconai.parameters import Parameters
from reconai.data.data import get_dataset_batchers, prepare_input, prepare_input_as_variable, append_to_file
from reconai.model.dnn_io import from_tensor_format
from reconai.utils.graph import print_acceleration_train_loss, print_acceleration_validation_loss, print_loss_progress,\
    print_prediction_error, print_full_prediction_sequence, print_loss_comparison_graphs, print_iterations, print_end_of_epoch
from reconai.utils.metric import complex_psnr
from reconai.model.model_pytorch import CRNNMRI
from reconai.model.module import Module

from os.path import join
import numpy as np

import reconai.utils.metric



def test_accelerations(args: Box):
    accelerations = [1, 2, 4, 8, 12, 16, 32]
    # accelerations = [32, 64]

    results = []
    for acceleration in accelerations:
        args['acceleration_factor'] = acceleration
        train_results = train(args)
        fold0, train_err, val_err = train_results[0]
        results.append((acceleration, train_err, val_err))

    print_acceleration_train_loss(results, args.num_epoch, args.loss, args.out_dir / f'acceleration_{args.date}')

    print_acceleration_validation_loss(results, args.num_epoch, args.loss, args.out_dir / f'acceleration_{args.date}')


def train(params: Parameters) -> List[tuple[int, List[int], List[int]]]:
    if not torch.cuda.is_available() and not params.debug:
        raise Exception('Can only run in Cuda')

    num_epoch = 3 if params.debug else params.config.train.epochs
    n_folds = params.config.train.folds if params.config.train.folds > 2 else 1

    # Configure directory info
    save_dir: Path = params.out_dir / params.date_name
    save_dir.mkdir(parents=True)
    logging.info(f"saving model to {save_dir.absolute()}")

    # Specify network
    network = CRNNMRI(n_ch=1, nc=2 if params.debug else 5).cuda()
    optimizer = optim.Adam(network.parameters(), lr=float(params.config.train.lr), betas=(0.5, 0.999))

    criterion = torch.nn.MSELoss().cuda()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_val_batcher, test_batcher = get_dataset_batchers(params.in_dir, params.config.data.slices)

    results = []
    logging.info(f'started {n_folds}-fold training at {datetime.datetime.now()}')
    for fold in range(n_folds):
        fold_dir = save_dir / f'fold_{fold}'

        graph_train_err, graph_val_err = [], []
        for epoch in range(num_epoch):
            t_start = time.time()

            # train, validate, test = get_dataset_batchers(args, data, n_folds, fold)
            train_err, train_batches = 0, 0
            for im in train_val_batcher.items_fold(fold, 5, validation=False):
                logging.debug(f"batch {train_batches}")
                im_u, k_u, mask, gnd = prepare_input_as_variable(im, args.seed, params.config.train.undersampling, args.equal_images) # TODO: Fix with params


                optimizer.zero_grad(set_to_none=True)
                rec, full_iterations = network(im_u, k_u, mask, gnd)
                loss = criterion(rec, gnd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1

                if params.debug and train_batches == 2:
                    break
            logging.info(f"completed {train_batches} train batches")

            validate_err, validate_batches = 0, 0
            network.eval()
            with torch.no_grad():
                for im in train_val_batcher.items_fold(fold, 5, validation=True):
                    logging.debug(f"batch {validate_batches}")
                    im_u, k_u, mask, gnd = prepare_input_as_variable(im, args.seed, params.config.train.undersampling, args.equal_images) # TODO: Fix with params

                    pred, full_iterations = network(im_u, k_u, mask, gnd, test=True)
                    err = criterion(pred, gnd)

                    validate_err += err.item()
                    validate_batches += 1

                    if params.debug and validate_batches == 2:
                        break
            logging.info(f"completed {validate_batches} validate batches")

            vis, iters, base_psnr, test_psnr, test_batches = [], [], 0, 0, 0
            with torch.no_grad():
                for im in test_batcher.items():
                    logging.debug(f"batch {test_batches}")
                    im_und, k_und, mask, im_gnd = prepare_input(im, args.seed, params.config.train.undersampling, args.equal_images) # TODO: Fix with params
                    im_u = Variable(im_und.type(Module.TensorType))
                    k_u = Variable(k_und.type(Module.TensorType))
                    mask = Variable(mask.type(Module.TensorType))

                    pred, full_iterations = network(im_u, k_u, mask, im_gnd, test=True)

                    for im_i, und_i, pred_i in zip(im,
                                                   from_tensor_format(im_und.numpy()),
                                                   from_tensor_format(pred.data.cpu().numpy())):
                        base_psnr += complex_psnr(im_i, und_i, peak='max')
                        test_psnr += complex_psnr(im_i, pred_i, peak='max')

                    vis.append((from_tensor_format(im_gnd.numpy())[0],
                                from_tensor_format(pred.data.cpu().numpy())[0],
                                from_tensor_format(im_und.numpy())[0],
                                0))
                    iters.append((from_tensor_format(im_gnd.numpy())[-1],
                                  full_iterations))

                    test_batches += 1
                    if params.debug and test_batches == 2:
                        break
            logging.info(f"completed {test_batches} test batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches
            base_psnr /= (test_batches * params.batch_size)
            test_psnr /= (test_batches * params.batch_size)

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epoch}',
                               f'\ttime: {t_end - t_start} s',
                               f'\ttraining loss:\t\t{train_err}',
                               f'\tvalidation loss:\t\t{validate_err}',
                               f'\tbase PSNR:\t\t\t{base_psnr}',
                               f'\ttest PSNR:\t\t\t{test_psnr}'
                               ])
            logging.info(stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            print_loss_progress(graph_train_err, graph_val_err, fold_dir, params.config.train.loss)

            if epoch % 5 == 0 or epoch > num_epoch - 5:
                name = f'{params.name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)

                # Loop through vis and gather mean, min and max
                result_ssim = []
                for i, (gnd, pred, und, seg) in enumerate(vis):
                    result_ssim.append(reconai.utils.metric.ssim(gnd[-1], pred[-1]))

                # print_full_prediction_sequence(epoch_dir, vis, name, validate_err,
                #                                params.config.data.slices, params.config.train.undersampling)
                arr = np.asarray(result_ssim)

                idx_min = arr.argmin()
                idx_mean = (np.abs(arr - np.mean(arr))).argmin()
                idx_max = arr.argmax()

                print_end_of_epoch(epoch_dir, [vis[idx_min]], f'{name}_worst', validate_err, result_ssim[idx_min],
                                   args.sequence_len, args.acceleration_factor, iters[idx_min][0], iters[idx_min][1], 10)
                print_end_of_epoch(epoch_dir, [vis[idx_mean]], f'{name}_average', validate_err,
                                   result_ssim[idx_mean], args.sequence_len, args.acceleration_factor,
                                   iters[idx_mean][0], iters[idx_mean][1], 10)
                print_end_of_epoch(epoch_dir, [vis[idx_max]], f'{name}_best', validate_err,
                                   result_ssim[idx_max], args.sequence_len, args.acceleration_factor,
                                   iters[idx_max][0], iters[idx_max][1], 10)

                torch.save(network.state_dict(), epoch_dir / npz_name)
                with open(epoch_dir / f'{name}.log', 'w') as log:
                    log.write(stats)

                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')
            append_to_file(fold_dir, params.config.train.undersampling, fold, epoch, train_err, validate_err)

            if epoch < 45:
                scheduler.step()

        results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.datetime.now()}')

    return results
