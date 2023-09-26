import os
from pathlib import Path

import click
import wandb
import nnunetv2

from .train import train
from .test import test
from .parameters import TrainParameters, TestParameters, Parameters
from .data import prepare_nnunet2, validate_nnunet2_dir


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
def reconai_train(in_dir: Path, out_dir: Path, config: Path, wandb_api: str):
    params = TrainParameters(in_dir, out_dir, config)
    setup_wandb(params, api_key=wandb_api, group='train_debug' if params.meta.debug else 'train')
    train(params)
    wandb.finish()


@cli.command(name='test')
@click.option('--in_dir', type=Path, required=True, help='Test data directory')
@click.option('--model_dir', type=Path, required=True, help='Trained model directory')
@click.option('--nnunet_dir', type=Path, required=False, help='Trained nnunet directory')
@click.option('--model_name', type=str, required=False, help='Use a specific model by name')
def reconai_test(in_dir: Path, model_dir: Path, nnunet_dir: Path, model_name: str):
    params = TestParameters(in_dir, model_dir, nnunet_dir, model_name)
    test(params)


@cli.command(name='train_segmentation')
@click.option('--in_dir', type=Path, required=True,
              help='Images data directory for training OR nnUNet_raw.')
@click.option('--annotation_dir', type=Path, required=False,
              help='Annotations data directory for training.')
@click.option('--out_dir', type=Path, required=True,
              help='Trained model output directory.')
def reconai_train_segmentation(in_dir: Path, annotation_dir: Path, out_dir: Path):
    if annotation_dir:
        prepare_nnunet2(in_dir, annotation_dir, out_dir)
    else:
        validate_nnunet2_dir(in_dir)


# @cli.command(name='eval')
# @click.option('--in_dir', type=Path, required=True)
# @click.option('--out_dir', type=Path, required=True)
# @click.option('--config', type=Path, required=True)
# @click.option('--wandb_api', type=str, required=True)
# @click.option('--debug', is_flag=True, default=False)
# def evaluate_models(in_dir: Path, out_dir: Path, config: Path, wandb_api: str, debug: bool):
#     params = Parameters(in_dir, out_dir, config, debug)
#     setup_wandb(params, api_key=wandb_api, group='eval_debug' if debug else 'eval')
#     # evaluate(params)
#     wandb.finish()


def setup_wandb(params: Parameters, api_key: str, group: str = ''):
    if api_key:
        wandb.login(key=api_key)
        wandb.init(project='FastIMRI-ReconAI', group=group, name=params.meta.name, config=params.as_dict())
        wandb.define_metric('epoch')


if __name__ == '__main__':
    cli()
