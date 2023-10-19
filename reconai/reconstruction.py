import json
import subprocess
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as torch_optim
import torch.utils.data as torch_data
import wandb
import SimpleITK as sitk
from PIL import Image

from reconai import version
from reconai.data import preprocess_as_variable, DataLoader, Dataset
from reconai.evaluation import Evaluation
from reconai.model.model_pytorch import CRNNMRI
from reconai.parameters import TestParameters, TrainParameters
from reconai.segmentation import test as segment, nnunet2_verify_results_dir
from reconai.print import print_log
from reconai.random import rng


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def view(x: torch.Tensor):
    plt.imshow(x.cpu(), cmap='gray')
    plt.show()


def train(params: TrainParameters):
    print_log(f'reconai version {version}', params.meta.name)
    print(str(params))

    if not torch.cuda.is_available():
        raise Exception('Can only run in Cuda')

    dataset_full = Dataset(params.in_dir)
    if params.data.normalize <= 0:
        for sample in DataLoader(dataset_full, shuffle=False, batch_size=1000):
            params.data.normalize = float(np.percentile(sample['data'], 95))
            print(f'data:\n  normalize: {params.data.normalize}')
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
                for i in range(len(batch['paths'])):
                    j = i + 1
                    optimizer.zero_grad()
                    pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j])
                    evaluator_train.calculate(pred, gnd[i:j])
                    evaluator_train.loss.backward()

                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)
                    optimizer.step()

            network.eval()
            with torch.no_grad():
                for batch in dataloader_validate:
                    im_u, k_u, mask, gnd = preprocess_as_variable(batch['data'], params.data.undersampling)
                    for i in range(len(batch['paths'])):
                        j = i + 1
                        evaluator_validate.start_timer()
                        pred, _ = network(im_u[i:j], k_u[i:j], mask[i:j], test=True)
                        evaluator_validate.calculate(pred, gnd[i:j])

            epoch_end = datetime.now()

            model = network.state_dict()
            stats = {'fold': fold,
                     'epoch': epoch,
                     'epoch_time': (epoch_end - epoch_start).total_seconds(),
                     'loss_train': evaluator_train.criterion_stats('loss'),
                     'loss_validate': evaluator_validate.criterion_stats('loss'),
                     'ssim_validate': evaluator_validate.criterion_stats('ssim'),
                     'mse_validate': evaluator_validate.criterion_stats('mse'),
                     'time_validate': evaluator_validate.criterion_stats('time')
                     }
            for key in [k for k, v in stats.items() if isinstance(v, tuple)]:
                value = stats.pop(key)
                stats |= {f'{key}_min': value[0], f'{key}_mean': value[1], f'{key}_max': value[2]}

            wandb.log(stats)
            print_log(json.dumps(stats, indent=2))

            def save_model(path: Path, model_stats: dict):
                torch.save(model, path)
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(model_stats, f, indent=4)

            save_model(params.out_dir / f'reconai_{fold}.npz', stats)

            _, validate_loss, _ = evaluator_validate.criterion_stats('loss')
            if validate_loss <= validate_loss_best:
                validate_loss_best = validate_loss
                save_model(params.out_dir / f'reconai_{fold}_best.npz', stats)

            if params.train.lr_decay_end == -1 or epoch < params.train.lr_decay_end:
                scheduler.step()

    end = datetime.now()
    print_log(f'completed training in {(end - start).total_seconds()} seconds, at {end}')


def test(params: TestParameters, nnunet_dir: Path, annotations_dir: Path):
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

    nnunet_out: Path | None = None
    nnunet_enabled = nnunet_dir and annotations_dir
    if nnunet_enabled:
        print_log('nnUNet enabled')
        assert nnunet_dir, 'nnUNet base directory is not set, but annotations directory is'
        assert annotations_dir, 'annotations directory is not set, but nnUNet base directory is'
        nnunet2_verify_results_dir(nnunet_dir)

        if annotations_dir.exists():
            nnunet2_images = {file.stem[:-5] + file.suffix for file in params.in_dir.iterdir() if file.suffix == '.mha'}
            nnunet2_annotations = {file.name for file in annotations_dir.iterdir() if file.suffix == '.mha'}
            assert nnunet2_images == nnunet2_annotations, f'{params.in_dir} and {annotations_dir} contents do not match'
            print_log(f'{params.in_dir} and {annotations_dir} contents match')
        else:
            print(f'segmenting {params.in_dir} to {annotations_dir}...')
            segment(params.in_dir, nnunet_dir, annotations_dir)
            print_log('nnUNet complete; run this command again with exact same parameters to use')
            return

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
    if nnunet_enabled:
        (nnunet_out := params.out_dir / 'nnunet').mkdir()

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

                if nnunet_enabled:
                    sitk_image = sitk.GetImageFromArray(pred.squeeze().cpu().numpy().transpose(2, 0, 1))
                    sitk.WriteImage(sitk_image, nnunet_out / path.name)

                for s in range(params.data.sequence_length):
                    t = s + 1
                    pred_single = pred[..., s:t]
                    evaluator_single.calculate(pred_single, gnd[i:j, ..., s:t], f'{path.stem}_slice_{s}')
                    img = Image.fromarray((pred_single.squeeze() * 255).byte().cpu().numpy())
                    img.save(params.out_dir / f'{path.stem}_{s}.png')

    del network
    torch.cuda.empty_cache()

    stats = {'loss_test': evaluator.criterion_stats('loss'),
             'ssim_test': evaluator.criterion_stats('ssim'),
             'mse_test': evaluator.criterion_stats('mse'),
             'time_test': evaluator.criterion_stats('time'),
             'dataset_test': evaluator_single.criterion_value_per_key}
    for key in [k for k, v in stats.items() if isinstance(v, tuple)]:
        value = stats.pop(key)
        stats |= {f'{key}_min': value[0], f'{key}_mean': value[1], f'{key}_max': value[2]}

    if nnunet_enabled:
        print_log('calculating DICE scores...')
        segment(nnunet_out, nnunet_dir, nnunet_out)

        gnd_segmentations = {a.name: a for a in annotations_dir.iterdir() if a.suffix == '.mha'}
        pred_segmentations = {a.name: a for a in nnunet_out.iterdir() if a.suffix == '.mha' and not a.stem.endswith('_0000')}

        assert gnd_segmentations.keys() == pred_segmentations.keys()
        for name, gnd_path in gnd_segmentations.items():
            pred_path = pred_segmentations[name]
            gnd = sitk.GetArrayFromImage(sitk.ReadImage(gnd_path.as_posix()))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path.as_posix()))
            evaluator.calculate_dice(pred, gnd, key=f'{name[:-4]}_0000')

        stats['dice_test'] = evaluator.criterion_stats('dice')
    stats['dataset_test'] |= evaluator.criterion_value_per_key

    with open(params.out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print_log(f'complete; test results are in {params.out_dir}')
