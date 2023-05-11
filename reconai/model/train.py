#!/usr/bin/env python
from __future__ import print_function, division

import datetime
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from typing import List
from pathlib import Path
import logging
import numpy as np

from reconai.parameters import Parameters
from reconai.data.data import get_dataset_batchers, prepare_input, prepare_input_as_variable, append_to_file
from reconai.model.dnn_io import from_tensor_format
from reconai.utils.graph import print_acceleration_train_loss, print_acceleration_validation_loss, print_loss_progress,\
    print_end_of_epoch
from reconai.utils.metric import complex_psnr
from reconai.model.model_pytorch import CRNNMRI
from reconai.model.module import Module
from piqa import SSIM

import reconai.utils.metric

# TODO: herschrijven zodat het werkt met parameters uit yaml
# def test_accelerations(args: Box):
#     accelerations = [1, 2, 4, 8, 12, 16, 32]
#     # accelerations = [32, 64]
#
#     results = []
#     for acceleration in accelerations:
#         args['acceleration_factor'] = acceleration
#         train_results = train(args)
#         fold0, train_err, val_err = train_results[0]
#         results.append((acceleration, train_err, val_err))
#
#     print_acceleration_train_loss(results, args.num_epoch, args.loss, args.out_dir / f'acceleration_{args.date}')
#
#     print_acceleration_validation_loss(results, args.num_epoch, args.loss, args.out_dir / f'acceleration_{args.date}')


def train(params: Parameters) -> List[tuple[int, List[int], List[int]]]:
    if not torch.cuda.is_available() and not params.debug:
        raise Exception('Can only run in Cuda')

    num_epochs = 3 if params.debug else params.config.train.epochs
    n_folds = params.config.train.folds if params.config.train.folds > 2 else 1
    sequence_length = params.config.data.sequence_length
    undersampling = params.config.train.undersampling
    iterations = params.config.model.iterations

    # Configure directory info
    logging.info(f"saving model to {params.out_dir.absolute()}")

    # Specify network
    network = CRNNMRI(nc=2 if params.debug else iterations,
                      nf=params.config.model.filters).cuda()
    optimizer = optim.Adam(network.parameters(), lr=float(params.config.train.lr), betas=(0.5, 0.999))

    if params.config.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.config.train.loss.ssim == 1:
        # criterion = SSIMLoss().cuda()
        criterion = SSIM(n_channels=sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.config.train.lr_gamma)

    train_val_batcher, test_batcher_equal, test_batcher_non_equal = get_dataset_batchers(params)

    results = []
    logging.info(f'started {n_folds}-fold training at {datetime.datetime.now()}')
    for fold in range(n_folds):
        fold_dir = params.out_dir / f'fold_{fold}'

        graph_train_err, graph_val_err = [], []
        for epoch in range(num_epochs):
            t_start = time.time()

            network.train()
            train_err, train_batches = 0, 0
            for im in train_val_batcher.items_fold(fold, 5, validation=False):
                logging.debug(f"batch {train_batches}")
                im_u, k_u, mask, gnd = prepare_input_as_variable(im,
                                                                 params.config.train.mask_seed,
                                                                 params.config.train.undersampling,
                                                                 params.config.train.equal_masks)

                optimizer.zero_grad(set_to_none=True)
                rec, full_iterations = network(im_u, k_u, mask, gnd)
                # loss = criterion(rec.permute(0,1,4,2,3), gnd.permute(0,1,4,2,3))
                if params.config.train.loss.mse == 1:
                    loss = criterion(rec, gnd)
                elif params.config.train.loss.ssim == 1:
                    loss = 1 - criterion(rec.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])
                    # crit2 = SSIMLoss().cuda()
                    # loss2 = crit2(rec, gnd)
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
                    im_u, k_u, mask, gnd = prepare_input_as_variable(im,
                                                                     params.config.train.mask_seed,
                                                                     params.config.train.undersampling,
                                                                     params.config.train.equal_masks)

                    pred, full_iterations = network(im_u, k_u, mask, gnd, test=True)
                    if params.config.train.loss.mse == 1:
                        err = criterion(pred, gnd)
                    elif params.config.train.loss.ssim == 1:
                        err = 1 - criterion(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])

                    validate_err += err.item()
                    validate_batches += 1

                    if params.debug and validate_batches == 2:
                        break
            logging.info(f"completed {validate_batches} validate batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epochs}',
                               f'\ttime: {t_end - t_start} s',
                               f'\ttraining loss:\t\t{train_err}',
                               f'\tvalidation loss:\t\t{validate_err}',
                               ])
            logging.info(stats)

            if epoch % 5 == 0 or epoch > num_epochs - 5:
                name = f'{params.name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), epoch_dir / npz_name)
                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')

                if epoch % 10 == 0 or epoch > num_epochs - 5:
                    run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, epoch_dir,
                                            name, train_err, validate_err, stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            print_loss_progress(graph_train_err, graph_val_err, fold_dir, params.config.train.loss)
            append_to_file(fold_dir, undersampling, fold, epoch, train_err, validate_err)

            if params.config.train.stop_lr_decay == -1 or epoch < params.config.train.stop_lr_decay:
                scheduler.step()

        results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.datetime.now()}')

    return results

def run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params,
                            epoch_dir, name, train_err, validate_err, stats: str = ''):
    sequence_length = params.config.data.sequence_length
    undersampling = params.config.train.undersampling
    iterations = params.config.model.iterations

    vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e = \
        run_testset(network, test_batcher_equal, params, equal_mask=True)
    vis_ne, iters_ne, base_psnr_ne, test_psnr_ne, test_batches_ne = \
        run_testset(network, test_batcher_non_equal, params, equal_mask=True)
    vis_nenm, iters_nenm, base_psnr_nenm, test_psnr_nenm, test_batches_nenm = \
        run_testset(network, test_batcher_non_equal, params, equal_mask=False)
    logging.info(f"completed {test_batches_e} test batches")

    base_psnr_e /= (test_batches_e * params.batch_size)
    base_psnr_ne /= (test_batches_ne * params.batch_size)
    base_psnr_nenm /= (test_batches_nenm * params.batch_size)
    test_psnr_e /= (test_batches_e * params.batch_size)
    test_psnr_ne /= (test_batches_ne * params.batch_size)
    test_psnr_nenm /= (test_batches_nenm * params.batch_size)

    stats2 = '\n'.join([f'\tbase PSNR equal images:\t\t\t{base_psnr_e}',
                        f'\tbase PSNR non-equal images:\t\t\t{base_psnr_ne}',
                        f'\tbase PSNR non-equal images n-masks:\t\t\t{base_psnr_nenm}',
                        f'\ttest PSNR equal images:\t\t\t{test_psnr_e}',
                        f'\ttest PSNR non-equal images:\t\t\t{test_psnr_ne}',
                        f'\ttest PSNR non-equal images n-masks:\t\t\t{test_psnr_nenm}'])
    logging.info(stats2)

    # Loop through vis and gather mean, min and max SSIM to print each
    ssim_e, min_e, mean_e, max_e = print_eoe(vis_e, epoch_dir, f'{name}_equal', validate_err,
                                             sequence_length, undersampling, iters_e, iterations)
    ssim_ne, min_ne, mean_ne, max_ne = print_eoe(vis_ne, epoch_dir, f'{name}_nonequal', validate_err,
                                                 sequence_length, undersampling, iters_ne, iterations)
    ssim_nenm, min_nenm, mean_nenm, max_nenm = print_eoe(vis_nenm, epoch_dir, f'{name}_nonequal_nmasks',
                                                         validate_err, sequence_length, undersampling,
                                                         iters_nenm, iterations)

    def ft(in_float: float) -> str:
        return format(in_float, '.4f').replace('.', ',')

    with open(epoch_dir / f'{name}.log', 'w') as log:
        log.write(stats)
        log.write(stats2)
        log.write('\n\n SSIM - Equal Images \n')
        log.write(f'MIN: {min_e} | MEAN: {mean_e} | MAX: {max_e} \n\n')
        log.write(np.array2string(ssim_e, separator=','))
        log.write('\n\n SSIM - Non-Equal Images \n')
        log.write(f'MIN: {min_ne} | MEAN: {mean_ne} | MAX: {max_ne} \n\n')
        log.write(np.array2string(ssim_ne, separator=','))
        log.write('\n\n SSIM - Non-Equal Images N-Masks\n')
        log.write(f'MIN: {min_nenm} | MEAN: {mean_nenm} | MAX: {max_nenm} \n\n')
        log.write(np.array2string(ssim_nenm, separator=','))
        log.write('\n\n train_err validate_err min_e mean_e max_e ... ne ... nenm')
        log.write(f"\n\n\n{ft(train_err)}\t {ft(validate_err)}"
                  f"\t {ft(min_e)}\t {ft(mean_e)}\t{ft(max_e)}"
                  f"\t {ft(min_ne)}\t {ft(mean_ne)}\t{ft(max_ne)}"
                  f"\t {ft(min_nenm)}\t {ft(mean_nenm)}\t{ft(max_nenm)}")

def run_testset(network, batcher, params, equal_mask: bool):
    vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e = [], [], 0, 0, 0

    with torch.no_grad():
        for im in batcher.items():
            logging.debug(f"batch {test_batches_e}")
            im_und, k_und, mask, im_gnd = prepare_input(im,
                                                        params.config.train.mask_seed,
                                                        params.config.train.undersampling,
                                                        equal_mask=equal_mask)
            im_u = Variable(im_und.type(Module.TensorType))
            k_u = Variable(k_und.type(Module.TensorType))
            mask = Variable(mask.type(Module.TensorType))

            pred, full_iterations = network(im_u, k_u, mask, im_gnd, test=True)

            for im_i, und_i, pred_i in zip(im,
                                           from_tensor_format(im_und.numpy()),
                                           from_tensor_format(pred.data.cpu().numpy())):
                base_psnr_e += complex_psnr(im_i, und_i, peak='max')
                test_psnr_e += complex_psnr(im_i, pred_i, peak='max')

            vis_e.append((from_tensor_format(im_gnd.numpy())[0],
                          from_tensor_format(pred.data.cpu().numpy())[0],
                          from_tensor_format(im_und.numpy())[0],
                          0))
            iters_e.append((from_tensor_format(im_gnd.numpy())[-1],
                            full_iterations))

            test_batches_e += 1
            if params.debug and test_batches_e == 2:
                break
    return vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e



def print_eoe(vis, epoch_dir, name, validate_err, sequence_length, undersampling, iters, iterations):
    result_ssim = []
    for i, (gnd, pred, und, seg) in enumerate(vis):
        result_ssim.append(reconai.utils.metric.ssim(gnd[-1], pred[-1]))

    arr = np.asarray(result_ssim)

    idx_min = arr.argmin()
    idx_mean = (np.abs(arr - np.mean(arr))).argmin()
    idx_max = arr.argmax()

    print_end_of_epoch(epoch_dir, [vis[idx_min]], f'{name}_worst', validate_err, result_ssim[idx_min],
                       sequence_length, undersampling, iters[idx_min][0], iters[idx_min][1], iterations)
    print_end_of_epoch(epoch_dir, [vis[idx_mean]], f'{name}_average', validate_err,
                       result_ssim[idx_mean], sequence_length, undersampling, iters[idx_mean][0],
                       iters[idx_mean][1], iterations)
    print_end_of_epoch(epoch_dir, [vis[idx_max]], f'{name}_best', validate_err,
                       result_ssim[idx_max], sequence_length, undersampling, iters[idx_max][0],
                       iters[idx_max][1], iterations)
    return arr, result_ssim[idx_min], result_ssim[idx_mean], result_ssim[idx_max]
