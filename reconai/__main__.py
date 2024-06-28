from pathlib import Path

import click
import wandb

from reconai import version
from .parameters import ModelTrainParameters, ModelParameters
from .reconstruction import train as train_reconstruction, reconstruct
from .segmentation import (train as train_segmentation,
                           nnunet2_prepare_nnunet, nnunet2_find_best_configuration, nnUNet_dataset_name)
from .test import test


@click.group()
@click.version_option(version)
def cli():
    pass


@cli.command(name='train_reconstruction')
@click.option('--in_dir', type=Path, required=True,
              help='Training data directory.')
@click.option('--out_dir', type=Path, required=True,
              help='Trained model output directory.')
@click.option('--config', type=Path, required=False,
              help='Config .yaml file. If undefined, use the config_debug.yaml file.')
@click.option('--wandb_api', type=str, required=False, help='wandb api key')
@click.option('--retry', type=bool, default=False, help='Retry fold if loss explodes.')
def reconai_train_reconstruction(in_dir: Path, out_dir: Path, config: Path, wandb_api: str, retry: bool):
    params = ModelTrainParameters(in_dir, out_dir, config)
    if wandb_api:
        wandb.login(key=wandb_api)
        wandb.init(project='FastIMRI-ReconAI',
                   group='train_debug' if params.meta.debug else 'train',
                   name=out_dir.name,
                   config=params.as_dict())
        wandb.define_metric('epoch')

    train_reconstruction(params, bool(wandb_api), retry)
    if wandb_api:
        wandb.finish()


@cli.command(name='reconstruct')
@click.option('--in_dir', type=Path, required=True,
              help='Test data directory.')
@click.option('--model_dir', type=Path, required=True,
              help='Trained model directory.')
@click.option('--out_dir', type=Path, required=True,
              help='Results directory.')
@click.option('--out_png', type=bool, help='Also output as images.')
def reconai_reconstruct(in_dir: Path, model_dir: Path, out_dir: Path, out_png: bool):
    assert in_dir != out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with reconstruct(ModelParameters(in_dir, model_dir), out_png) as recon:
        for file in in_dir.iterdir():
            if file.suffix == '.mha':
                recon(file, out_dir / file.name)


@cli.command(name='train_segmentation')
@click.option('--in_dir', type=Path, required=True,
              help='Images data directory for training OR directory containing nnUNet directories.')
@click.option('--annotations_dir', type=Path, required=False,
              help='Annotations data directory for training (if in_dir is not nnUNet).')
@click.option('--out_dir', type=Path, required=False,
              help='Output directory to contain nnUNet directories. (if in_dir is not nnUNet)')
@click.option('--folds', type=int, required=False, default=5,
              help='Number of folds.')
@click.option('--gpus', type=int, required=False, default=1,
              help='Number of GPUs')
@click.option('--sync_dir', type=Path, required=True,
              help='Sync out_dir to sync_dir')
@click.option('--debug', is_flag=True, hidden=True, default=False)
def reconai_train_segmentation(in_dir: Path, annotations_dir: Path, out_dir: Path, folds: int, gpus: int, sync_dir: Path, debug: bool = False):
    train_segmentation(in_dir, annotations_dir, out_dir, sync_dir, folds, gpus, debug)


@cli.command(name='test')
@click.option('--in_dir', type=Path, required=True,
              help='Test data directory.')
@click.option('--model_dir', type=Path, required=True,
              help='Trained model directory.')
@click.option('--nnunet_dir', type=Path, required=False,
              help='Directory containing nnUNet directories.')
@click.option('--annotations_dir', type=Path, required=False,
              help='Annotation data directory.')
@click.option('--model_name', type=str, required=False,
              help='Use a specific model by name')
@click.option('--tag', type=str, required=False,
              help='Specify a tag to name this test.')
@click.option('--debug', is_flag=True, hidden=True, default=False)
def reconai_test(in_dir: Path, model_dir: Path, nnunet_dir: Path, annotations_dir: Path, model_name: str, tag: str, debug: bool = False):
    assert not ((nnunet_dir is None) ^ (annotations_dir is None)), '--nnunet_dir AND --annotations_dir need be defined'
    params = ModelParameters(in_dir, model_dir, model_name, tag)
    test(params, nnunet_dir, annotations_dir, debug)


@cli.command(name='test_find_configuration')
@click.option('--nnunet_dir', type=Path, required=True,
              help='Directory containing nnUNet directories.')
@click.option('--debug', is_flag=True, hidden=True, default=False)
def reconai_test_find_configuration(nnunet_dir: Path, debug: bool = False):
    nnUNet_results = nnunet_dir / 'nnUNet_results'
    dataset_dir = nnUNet_results / nnUNet_dataset_name
    configs, folds = [], set()
    for config_dir in dataset_dir.iterdir():
        if config_dir.is_dir():
            configs.append(config_dir.name.split('__')[-1])
            folds = folds.union([fold_dir.name.split('_')[-1] for fold_dir in config_dir.iterdir() if
                          fold_dir.name.startswith('fold_')])

    nnunet2_prepare_nnunet(nnunet_dir)
    nnunet2_find_best_configuration(configs, list(folds), debug=debug)


if __name__ == '__main__':
    cli()
