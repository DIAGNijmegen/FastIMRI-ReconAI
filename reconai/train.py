import json
from datetime import datetime
from pathlib import Path
from typing import List, Callable

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
    criterion = criterion_func(params)

    print_log(f'trainable parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_full)} items',
              f'saving model data to {params.out_dir.resolve()}')
    params.mkoutdir()

    folds = params.train.folds
    start = datetime.now()
    print_log(f'starting {folds}-fold training at {start}')
    for fold, dataset in enumerate(torch_data.random_split(dataset_full, [len(dataset_full) // folds] * folds)):
        rng(params.data.seed)
        torch.manual_seed(params.data.seed)

        dataset_split = [1 / folds] * folds if folds > 1 else [0.8, 0.2]
        dataset_train, dataset_validate = torch_data.random_split(dataset, [dataset_split[0], sum(dataset_split[1:])])
        dataloader_train, dataloader_validate = (DataLoader(ds, batch_size=params.data.batch_size) for ds in (dataset_train, dataset_validate))

        validate_best = 0
        for epoch in range(params.train.epochs):
            epoch_start = datetime.now()

            network.train()
            train_loss = 0
            for batch in dataloader_train:
                im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                optimizer.zero_grad(set_to_none=True)
                for i in range(len(batch)):
                    j = i + 1
                    pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j])
                    loss: torch.Tensor = criterion(pred, gnd[i:j])
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

                    train_loss += loss.item()

            network.eval()
            validate_loss, validate_ssim, validate_mse = 0, 0, 0
            with torch.no_grad():
                for batch in dataloader_validate:
                    im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                    for i in range(len(batch)):
                        j = i + 1
                        pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)

                        validate_loss += criterion(pred, gnd[i:j]).item()
                        validate_ssim += 1 - criterion(pred, gnd[i:j], mse=0, ssim=1, dice=0).item()
                        validate_mse += criterion(pred, gnd[i:j], mse=1, ssim=0, dice=0).item()

            epoch_end = datetime.now()

            train_loss /= len(dataloader_train)
            validate_loss /= len(dataloader_validate)
            validate_ssim /= len(dataloader_validate)
            validate_mse /= len(dataloader_validate)

            model = network.state_dict()
            log = {'epoch': epoch,
                   'epoch_time': (epoch_end - epoch_start).total_seconds(),
                   'loss_train': train_loss,
                   'loss_validate': validate_loss,
                   'ssim_validate': validate_ssim,
                   'mse_validate': validate_mse}

            print_log(*[f'{key:<13}: {value:<20}' for key, value in log.items()])
            wandb.log(log)

            def save_model(path: Path):
                torch.save(model, path)
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(log, f, indent=4)

            save_model(params.out_dir / f'reconai_{fold}.npz')
            if validate_loss >= validate_best:
                validate_best = validate_loss
                save_model(params.out_dir / f'reconai_{fold}_best.npz')

            if params.train.lr_decay_end == -1 or epoch < params.train.lr_decay_end:
                scheduler.step()

    end = datetime.now()
    print_log(f'completed training in {(end - start).total_seconds()} seconds, at {end}')


def criterion_func(params: Parameters) -> Callable[..., torch.Tensor]:
    crit = {
        'mse': torch.nn.MSELoss().cuda(),
        'ssim': SSIM(n_channels=params.data.sequence_length).cuda(),
        'dice': lambda: 0
    }

    def calculate_loss(pred, gnd, mse: float = None, ssim: float = None, dice: float = None):
        weighted_loss: List[(float, torch.Tensor)] = []
        mse = params.train.loss.mse if mse is None else mse
        ssim = params.train.loss.ssim if ssim is None else ssim
        dice = params.train.loss.dice if dice is None else dice
        if mse > 0:
            weighted_loss.append((mse, crit['mse'](pred, gnd)))
        if ssim > 0:
            pred_permute, gnd_permute = pred.permute(0, 1, 4, 2, 3)[0], gnd.permute(0, 1, 4, 2, 3)[0]
            weighted_loss.append((ssim, 1 - crit['ssim'](pred_permute, gnd_permute)))
        if dice > 0:
            raise NotImplementedError("Only MSE or SSIM loss implemented")

        loss_sum = torch.tensor(0, device='cuda', dtype=gnd.dtype)
        for weight, value in weighted_loss:
            loss_sum += weight * value
        return loss_sum

    return calculate_loss
