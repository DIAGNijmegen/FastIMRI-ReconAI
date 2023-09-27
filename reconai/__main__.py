from pathlib import Path

import click
import wandb

from .parameters import TrainParameters, TestParameters, Parameters
from .reconstruction import train as train_reconstruction, test as test_reconstruction
from .segmentation import train as train_segmentation, test as test_segmentation


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--in_dir', type=Path, required=True,
              help='Training data directory.')
@click.option('--out_dir', type=Path, required=True,
              help='Trained model output directory.')
@click.option('--config', type=Path, required=False,
              help='Config .yaml file. If undefined, use a DEBUG .yaml file.')
@click.option('--wandb_api', type=str, required=True, help='wandb api key')
def reconai_train_reconstruction(in_dir: Path, out_dir: Path, config: Path, wandb_api: str):
    params = TrainParameters(in_dir, out_dir, config)
    setup_wandb(params, api_key=wandb_api, group='train_debug' if params.meta.debug else 'train')
    train_reconstruction(params)
    wandb.finish()


@cli.command(name='test')
@click.option('--in_dir', type=Path, required=True, help='Test data directory')
@click.option('--model_dir', type=Path, required=True, help='Trained model directory')
@click.option('--nnunet_dir', type=Path, required=False, help='Trained nnunet directory')
@click.option('--model_name', type=str, required=False, help='Use a specific model by name')
def reconai_test_reconstruction(in_dir: Path, model_dir: Path, nnunet_dir: Path, model_name: str):
    params = TestParameters(in_dir, model_dir, nnunet_dir, model_name)
    test_reconstruction(params)


@cli.command(name='train_segmentation')
@click.option('--in_dir', type=Path, required=True,
              help='Images data directory for training OR nnUNet_raw directory.')
@click.option('--annotation_dir', type=Path, required=False,
              help='Annotations data directory for training (if in_dir is not nnUNet_raw).')
@click.option('--out_dir', type=Path, required=False,
              help='Output directory to contain nnUNet_raw. (if in_dir is not nnUNet_raw)')
@click.option('--folds', type=int, required=False, default=5,
              help='Number of folds to train')
@click.option('--debug', is_flag=True, hidden=True, default=False)
def reconai_train_segmentation(in_dir: Path, annotation_dir: Path, out_dir: Path, folds: int, debug: bool = False):
    train_segmentation(in_dir, annotation_dir, out_dir, folds, debug)


@cli.command(name='test_segmentation')
@click.option('--in_dir', type=Path, required=True,
              help='Images data directory.')
@click.option('--nnunet_dir', type=Path, required=False,
              help='nnUNet_results directory.')
@click.option('--out_dir', type=Path, required=False,
              help='Output directory to contain inferences')
@click.option('--debug', is_flag=True, hidden=True, default=False)
def reconai_test_segmentation(in_dir: Path, nnunet_dir: Path, debug: bool = False):
    test_segmentation(in_dir, nnunet_dir, debug)


def setup_wandb(params: Parameters, api_key: str, group: str = ''):
    if api_key:
        wandb.login(key=api_key)
        wandb.init(project='FastIMRI-ReconAI', group=group, name=params.meta.name, config=params.as_dict())
        wandb.define_metric('epoch')


if __name__ == '__main__':
    cli()
