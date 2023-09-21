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
from reconai.parameters import TestParameters
from reconai.print import print_log
from reconai.rng import rng


def test(params: TestParameters):
    print_log(f'reconai version {version}', params.meta.name)
    assert version == params.meta.version

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    dataset_test = Dataset(params.in_dir, normalize=params.data.normalize)

    network = CRNNMRI(n_ch=params.model.channels,
                      nf=params.model.filters,
                      ks=params.model.kernelsize,
                      nc=params.model.iterations,
                      nd=params.model.layers,
                      bcrnn=params.model.bcrnn
                      ).cuda()

    evaluator = Evaluation(params)

    network.load_state_dict(torch.load(params.npz))
    network.eval()

    print_log(f'model parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_test)} items',
              f'saving results to {params.out_dir.resolve()}')
    params.mkoutdir()

    rng(params.data.seed)
    torch.manual_seed(params.data.seed)
    with torch.no_grad():
        dataloader_test = DataLoader(dataset_test, batch_size=params.data.batch_size)

        for batch in dataloader_test:
            im_u, k_u, mask, gnd = preprocess_as_variable(batch['data'], params.data.undersampling)
            paths = batch['path']
            for i in range(len(batch)):
                j = i + 1
                evaluator.start_timer()
                pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                evaluator.calculate(pred, gnd[i:j], paths[i])
                np.save((params.out_dir / paths[i]).with_suffix('.png'), pred)
                # save segmentation also

        stats = {'loss_test': evaluator['loss'],
                 'ssim_test': evaluator['ssim'],
                 'mse_test': evaluator['mse'],
                 'time_test': evaluator['time'],
                 'dataset': evaluator.paths}

        print_log(*[f'{key:<13}: {value:<20}' for key, value in stats.items()])
        wandb.log(stats)

        with open(params.out_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
