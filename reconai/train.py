import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as torch_optim
import torch.utils.data as torch_data
import wandb

from reconai import version
from reconai.criterion import Criterion
from reconai.data import preprocess_as_variable, DataLoader, Dataset
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import TrainParameters
from reconai.print import print_log
from reconai.rng import rng


def view(x: torch.Tensor):
    plt.imshow(x.cpu(), cmap='gray')
    plt.show()


def train(params: TrainParameters):
    print_log(f'reconai version {version}', params.meta.name)
    print(str(params))

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    dataset_full = Dataset(params.in_dir)
    if params.data.normalize == 0 or True:
        for sample in DataLoader(dataset_full, shuffle=False, batch_size=1000):
            params.data.normalize = float(np.percentile(sample, 95))
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
    criterion = Criterion(params)

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
        dataloader_train = DataLoader(dataset_train, batch_size=params.data.batch_size)
        dataloader_validate = DataLoader(dataset_validate, batch_size=params.data.batch_size)

        validate_best = 0
        for epoch in range(params.train.epochs):
            epoch_start = datetime.now()

            network.train()
            train_loss = []
            for batch in dataloader_train:
                p = dataloader_train.indices
                im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                optimizer.zero_grad(set_to_none=True)
                for i in range(len(batch)):
                    j = i + 1
                    pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j])
                    loss: torch.Tensor = criterion.weighted_loss(pred, gnd[i:j])
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

                    train_loss.append(loss.item())

            network.eval()
            validate_loss, validate_ssim, validate_mse = [], [], []
            with torch.no_grad():
                for batch in dataloader_validate:
                    im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
                    for i in range(len(batch)):
                        j = i + 1
                        pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)

                        validate_loss.append(criterion.weighted_loss(pred, gnd[i:j]).item())
                        validate_ssim.append(criterion.ssim(pred, gnd[i:j]).item())
                        validate_mse.append(criterion.mse(pred, gnd[i:j]).item())

            epoch_end = datetime.now()

            train_loss = sum(train_loss) / len(train_loss)
            validate_loss = sum(validate_loss) / len(validate_loss)
            validate_ssim = sum(validate_ssim) / len(validate_ssim)
            validate_mse = sum(validate_mse) / len(validate_mse)

            model = network.state_dict()
            log = {'fold': fold,
                   'epoch': epoch,
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
