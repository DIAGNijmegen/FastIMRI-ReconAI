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
from reconai.evaluation import Evaluation
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
            params.data.normalize = float(np.percentile(sample['data'], 95))
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

        validate_loss_best = np.inf
        for epoch in range(params.train.epochs):
            epoch_start = datetime.now()

            evaluator_train = Evaluation(params, loss_only=True)
            evaluator_validate = Evaluation(params)

            network.train()
            for batch in dataloader_train:
                im_u, k_u, mask, gnd = preprocess_as_variable(batch['data'], params.data.undersampling)
                optimizer.zero_grad(set_to_none=True)
                for i in range(len(batch)):
                    j = i + 1
                    pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j])
                    evaluator_train.calculate(pred, gnd[i:j])
                    evaluator_train.loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

            network.eval()
            with torch.no_grad():
                for batch in dataloader_validate:
                    im_u, k_u, mask, gnd = preprocess_as_variable(batch['data'], params.data.undersampling)
                    for i in range(len(batch)):
                        j = i + 1
                        evaluator_validate.start_timer()
                        pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                        evaluator_validate.calculate(pred, gnd[i:j])

            epoch_end = datetime.now()

            model = network.state_dict()
            stats = {'fold': fold,
                     'epoch': epoch,
                     'epoch_time': (epoch_end - epoch_start).total_seconds(),
                     'loss_train': evaluator_train['loss'],
                     'loss_validate': (validate_loss := evaluator_validate['loss']),
                     'ssim_validate': evaluator_validate['ssim'],
                     'mse_validate': evaluator_validate['mse'],
                     'time_validate': evaluator_validate['time']}

            print_log(*[f'{key:<13}: {value:<20}' for key, value in stats.items()])
            wandb.log(stats)

            def save_model(path: Path):
                torch.save(model, path)
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(stats, f, indent=4)

            save_model(params.out_dir / f'reconai_{fold}.npz')
            if validate_loss <= validate_loss_best:
                validate_loss_best = validate_loss
                save_model(params.out_dir / f'reconai_{fold}_best.npz')

            if params.train.lr_decay_end == -1 or epoch < params.train.lr_decay_end:
                scheduler.step()

    end = datetime.now()
    print_log(f'completed training in {(end - start).total_seconds()} seconds, at {end}')
