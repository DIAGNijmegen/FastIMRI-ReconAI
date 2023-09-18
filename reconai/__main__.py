from pathlib import Path

import click
import wandb

from .__version__ import __version__
from .train import train
# from .model.evaluate import evaluate
from .parameters import Parameters


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=False)
@click.option('--wandb_api', type=str, required=True)
@click.option('--debug', is_flag=True, default=False)
def reconai_train(in_dir: Path, out_dir: Path, config: Path, wandb_api: str, debug: bool):
    params = Parameters(in_dir, out_dir, config, debug)
    setup_wandb(params, api_key=wandb_api, group='train_debug' if debug else 'train')
    train(params)
    wandb.finish()


@cli.command(name='eval')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=True)
@click.option('--wandb_api', type=str, required=True)
@click.option('--debug', is_flag=True, default=False)
def evaluate_models(in_dir: Path, out_dir: Path, config: Path, wandb_api: str, debug: bool):
    params = Parameters(in_dir, out_dir, config, debug)
    setup_wandb(params, api_key=wandb_api, group='eval_debug' if debug else 'eval')
    # evaluate(params)
    wandb.finish()


def setup_wandb(params: Parameters, api_key: str, group: str = ''):
    if api_key:
        wandb.login(key=api_key)
        wandb.init(project='FastIMRI-ReconAI', group=group, name=params.name, config=params.as_dict())
        wandb.define_metric('epoch')


if __name__ == '__main__':
    cli()
