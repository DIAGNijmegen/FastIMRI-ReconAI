from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as torch_optim
import torch.utils.data as torch_data
import wandb
from piqa import SSIM

from reconai import version
from reconai.data import preprocess_as_variable, DataLoader, Dataset
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import Parameters
from reconai.print import print_log
from reconai.rng import rng


def view(x: torch.Tensor):
    plt.imshow(x.cpu(), cmap='gray')
    plt.show()


def train(params: Parameters):
    print_log(f'reconai version {version}', params.name)
    print(str(params))

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    dataset_full = Dataset(params.in_dir)
    if params.data.normalize == 0 or True:
        for sample in DataLoader(dataset_full, shuffle=False, batch_size=1000):
            params.data.normalize = np.percentile(sample, 95)
            break

    dataset_full.normalize = params.data.normalize

    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    optimizer = torch_optim.Adam(network.parameters(), lr=float(params.train.lr), betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.train.lr_gamma)

    if params.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    print_log(f'trainable parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_full)} items',
              f'saving model data to {params.out_dir.resolve()}')
    params.mkoutdir()

    folds = params.train.folds
    start = datetime.now()
    print_log(f'starting {folds}-fold training at {start}')
    for fold, dataset in enumerate(torch_data.random_split(dataset_full, [len(dataset_full) // folds] * folds)):
        fold_dir = params.out_dir / f'fold_{fold}'

        rng(params.data.seed)
        torch.manual_seed(params.data.seed)

        dataset_split = [1 / folds] * folds if folds > 1 else [0.8, 0.2]
        dataset_train, dataset_validate = torch_data.random_split(dataset, [dataset_split[0], sum(dataset_split[1:])])
        dataloader_train, dataloader_validate = (DataLoader(ds, batch_size=params.data.batch_size) for ds in (dataset_train, dataset_validate))

        for epoch in range(params.train.epochs):
            epoch_start = datetime.now()

            network.train()
            train_loss, train_batches = 0, 0
            for batch in dataloader_train:
                im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                batch_len = len(batch)
                optimizer.zero_grad(set_to_none=True)
                for i in range(batch_len):
                    j = i + 1
                    pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j])
                    loss: torch.Tensor = calculate_loss(params, criterion, pred, gnd[i:j])
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

                    train_loss += loss.item()
                train_batches += batch_len

            network.eval()
            validate_loss, validate_batches = 0, 0
            with torch.no_grad():
                for batch in dataloader_validate:
                    im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                    batch_len = len(batch)
                    for i in range(batch_len):
                        j = i + 1
                        pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                        loss: torch.Tensor = calculate_loss(params, criterion, pred, gnd[i:j])

                        validate_loss += loss.item()
                    validate_batches += batch_len

            epoch_end = datetime.now()

            train_loss /= train_batches
            validate_loss /= validate_batches

            log = {'epoch': epoch,
                   'epoch_time': (epoch_end - epoch_start).total_seconds(),
                   'loss_train': train_loss,
                   'loss_eval': validate_loss, }

            print_log(*[f'{key:<10}: {value:>20}' for key, value in log.items()])
            try:
                wandb.log(log)
            except wandb.errors.Error:
                pass



            # rework this part, save only best and last

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

    end = datetime.now()
    print_log(f'completed training in {(end - start).total_seconds()} seconds, at {end}')


def get_criterion(params: Parameters) -> torch.nn.Module:
    if params.train.loss.mse == 1:
        criterion = torch.nn.MSELoss().cuda()
    elif params.train.loss.ssim == 1:
        criterion = SSIM(n_channels=params.data.sequence_length).cuda()
    else:
        raise NotImplementedError("Only MSE or SSIM loss implemented")

    return criterion


def calculate_loss(params: Parameters, criterion, pred, gnd) -> torch.Tensor:
    loss: List[(float, torch.Tensor)] = []
    if params.train.loss.mse > 0:
        loss.append((params.train.loss.mse, criterion(pred, gnd)))
    if params.train.loss.ssim > 0:
        loss.append((params.train.loss.ssim, 1 - criterion(pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0])))
    if params.train.loss.dice > 0:
        raise NotImplementedError("Only MSE or SSIM loss implemented")
    loss_sum = torch.tensor(0, device='cuda', dtype=gnd.dtype)
    for weight, value in loss:
        loss_sum += weight * value
    return loss_sum


def log_epoch_stats_to_csv(fold_dir: Path, acceleration: float, fold: int, epoch: int, train_err: float,
                           val_err: float):
    with open(fold_dir / 'progress.csv', 'a+') as file:
        if epoch == 0:
            file.write('Acceleration, Fold, Epoch, Train error, Validation error \n')
        file.write(f'{acceleration}, {fold}, {epoch}, {train_err}, {val_err} \n')
