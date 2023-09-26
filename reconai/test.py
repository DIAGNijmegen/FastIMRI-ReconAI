import json
import re
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import wandb
from PIL import Image

from reconai import version
from reconai.data import preprocess_as_variable, DataLoader, Dataset
from reconai.evaluation import Evaluation
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import TestParameters
from reconai.print import print_log
from reconai.random import rng


def test(params: TestParameters):
    print_log(f'reconai version {version}', params.meta.name)
    version_re = r'(\d+)\.(\d+)\.(\d+)'
    version_r, params_version_r = re.match(version_re, version), re.match(version_re, params.meta.version)
    for v in [1, 2]:
        if int(version_v := version_r.group(v)) != int(params_version_v := params_version_r.group(v)):
            if v == 1:
                raise ImportError(f'major version mismatch: {version_v} != {params_version_v}')
            else:
                print(f'WARNING: minor version mismatch: {version_v} != {params_version_v}')

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
    params_1 = deepcopy(params)
    params_1.data.sequence_length = 1
    evaluator_single = Evaluation(params_1)

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
            paths = batch['paths']
            for i in range(len(paths)):
                j = i + 1
                path = Path(paths[i])
                evaluator.start_timer()
                pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                evaluator.calculate(pred, gnd[i:j], path.stem)

                for s in range(params.data.sequence_length):
                    t = s + 1
                    pred_single = pred[..., s:t]
                    evaluator_single.calculate(pred_single, gnd[i:j, ..., s:t], f'{path.stem}_{s}')
                    img = Image.fromarray((pred_single.squeeze() * 255).byte().cpu().numpy())
                    img.save(params.out_dir / f'{path.stem}_{s}.png')
                    # save segmentation also

        stats = {'loss_test': evaluator.criterion_value('loss'),
                 'ssim_test': evaluator.criterion_value('ssim'),
                 'mse_test': evaluator.criterion_value('mse'),
                 'time_test': evaluator.criterion_value('time'),
                 'dataset_test': evaluator.criterion_value_per_key | evaluator_single.criterion_value_per_key}

        with open(params.out_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
