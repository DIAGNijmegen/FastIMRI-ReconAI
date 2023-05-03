#!/usr/bin/env python
from __future__ import print_function, division

import datetime
import time

import ignite
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
# logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

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

            vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e = [], [], 0, 0, 0
            vis_ne, iters_ne, base_psnr_ne, test_psnr_ne, test_batches_ne = [], [], 0, 0, 0
            with torch.no_grad():
                for im in test_batcher_equal.items():
                    logging.debug(f"batch {test_batches_e}")
                    im_und, k_und, mask, im_gnd = prepare_input(im,
                                                                params.config.train.mask_seed,
                                                                params.config.train.undersampling,
                                                                params.config.train.equal_masks)
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
                for im in test_batcher_non_equal.items():
                    logging.debug(f"batch {test_batches_ne}")
                    im_und, k_und, mask, im_gnd = prepare_input(im,
                                                                params.config.train.mask_seed,
                                                                params.config.train.undersampling,
                                                                params.config.train.equal_masks)
                    im_u = Variable(im_und.type(Module.TensorType))
                    k_u = Variable(k_und.type(Module.TensorType))
                    mask = Variable(mask.type(Module.TensorType))

                    pred, full_iterations = network(im_u, k_u, mask, im_gnd, test=True)

                    for im_i, und_i, pred_i in zip(im,
                                                   from_tensor_format(im_und.numpy()),
                                                   from_tensor_format(pred.data.cpu().numpy())):
                        base_psnr_ne += complex_psnr(im_i, und_i, peak='max')
                        test_psnr_ne += complex_psnr(im_i, pred_i, peak='max')

                    vis_ne.append((from_tensor_format(im_gnd.numpy())[0],
                                  from_tensor_format(pred.data.cpu().numpy())[0],
                                  from_tensor_format(im_und.numpy())[0],
                                  0))
                    iters_ne.append((from_tensor_format(im_gnd.numpy())[-1],
                                    full_iterations))

                    test_batches_ne += 1
                    if params.debug and test_batches_ne == 2:
                        break
            logging.info(f"completed {test_batches_ne} test batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches
            base_psnr_e /= (test_batches_e * params.batch_size)
            base_psnr_ne /= (test_batches_ne * params.batch_size)
            test_psnr_e /= (test_batches_e * params.batch_size)
            test_psnr_ne /= (test_batches_ne * params.batch_size)

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epochs}',
                               f'\ttime: {t_end - t_start} s',
                               f'\ttraining loss:\t\t{train_err}',
                               f'\tvalidation loss:\t\t{validate_err}',
                               f'\tbase PSNR equal images:\t\t\t{base_psnr_e}',
                               f'\tbase PSNR non-equal images:\t\t\t{base_psnr_ne}',
                               f'\ttest PSNR equal images:\t\t\t{test_psnr_e}',
                               f'\ttest PSNR non-equal images:\t\t\t{test_psnr_ne}'
                               ])
            logging.info(stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            print_loss_progress(graph_train_err, graph_val_err, fold_dir, params.config.train.loss)

            if epoch % 5 == 0 or epoch > num_epochs - 5:
                name = f'{params.name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)

                # Loop through vis and gather mean, min and max SSIM to print each
                ssim_e, min_e, mean_e, max_e = print_eoe(vis_e, epoch_dir, f'{name}_equal', validate_err,
                                                         sequence_length, undersampling, iters_e, iterations)
                ssim_ne, min_ne, mean_ne, max_ne = print_eoe(vis_ne, epoch_dir, f'{name}_nonequal', validate_err,
                                                             sequence_length, undersampling, iters_ne, iterations)

                torch.save(network.state_dict(), epoch_dir / npz_name)
                with open(epoch_dir / f'{name}.log', 'w') as log:
                    log.write(stats)
                    log.write('\n\n SSIM - Equal Images \n')
                    log.write(f'MIN: {min_e} | MEAN: {mean_e} | MAX: {max_e} \n\n')
                    log.write(np.array2string(ssim_e, separator=','))
                    log.write('\n\n SSIM - Non-Equal Images \n')
                    log.write(f'MIN: {min_ne} | MEAN: {mean_ne} | MAX: {max_ne} \n\n')
                    log.write(np.array2string(ssim_ne, separator=','))

                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')
            append_to_file(fold_dir, undersampling, fold, epoch, train_err, validate_err)

            if params.config.train.stop_lr_decay == -1 or epoch < params.config.train.stop_lr_decay:
                scheduler.step()

        results.append((fold, graph_train_err, graph_val_err))
    logging.info(f'completed training at {datetime.datetime.now()}')

    return results

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

def eval_step(engine, batch):
    return batch
default_evaluator = ignite.engine.Engine(eval_step)

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_()

    def forward(self, x, y):
        # a = torchmetrics.functional.structural_similarity_index_measure(np.transpose(x.cpu(), (0, 1, 4, 2, 3)), np.transpose(y.cpu(), (0, 1, 4, 2, 3)))
        # ssim2 = kornia.metrics.ssim()
        # ssim = reconai.utils.metric.ssim(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), transpose=False)
        # t_start = time.time()
        # ssim2 = reconai.utils.metric.ssim_2(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy())
        # t_end = time.time()
        # logging.info(f'Time SSIM CPU {t_end - t_start}')
        # ssim3 = reconai.utils.metric.ssim_3(x.permute(0,1,4,2,3)[0], y.permute(0,1,4,2,3)[0])

        # t_start = time.time()
        metric = ignite.metrics.SSIM(data_range=1.0, device='cuda')
        metric.attach(default_evaluator, 'ssim')
        state = default_evaluator.run([[x.permute(0,1,4,2,3)[0], y.permute(0,1,4,2,3)[0]]])
        diff = torch.abs(torch.sub(state.metrics['ssim'], 1))
        diff.requires_grad = True
        # t_end = time.time()
        # logging.info(f'Time SSIM GPU {t_end - t_start}')
        return diff

