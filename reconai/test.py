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

    network.load_state_dict(torch.load(params.npz))
    network.eval()

    optimizer = torch_optim.Adam(network.parameters(), lr=float(params.train.lr), betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.train.lr_gamma)
    criterion = Criterion(params)

    print_log(f'model parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)}',
              f'data: {len(dataset_test)} items',
              f'saving results to {params.out_dir.resolve()}')
    params.mkoutdir()

    rng(params.data.seed)
    torch.manual_seed(params.data.seed)
    with torch.no_grad():
        dataloader_test = DataLoader(dataset_test, batch_size=params.data.batch_size)
        for batch in dataloader_test:
            im_u, k_u, mask, gnd = preprocess_as_variable(batch, params.data.undersampling)
            for i in range(len(batch)):
                j = i + 1
                pred, full_iterations = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)

                # validate_loss.append(criterion.weighted_loss(pred, gnd[i:j]).item())
                # validate_ssim.append(criterion.ssim(pred, gnd[i:j]).item())
                # validate_mse.append(criterion.mse(pred, gnd[i:j]).item())
                