from pathlib import Path
import logging

import click
import wandb as wdb
import shutil
from os.path import join

from .parameters import Parameters
from .model import train
from .model.evaluate import evaluate
from .__version__ import __version__


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=False)
@click.option('--wandb', type=str, required=False)
@click.option('--debug', is_flag=True, default=False)
def train_recon(in_dir: Path, out_dir: Path, config: Path, wandb: str, debug: bool):
    params = Parameters(in_dir, out_dir, config, debug)
    setup_logging(params)
    setup_wandb(wandb, params, group='debug' if debug else 'train')
    train(params)
    wdb.finish()


@cli.command(name='eval')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=True)
@click.option('--wandb', type=str, required=False)
@click.option('--debug', is_flag=True, default=False)
def evaluate_models(in_dir: Path, out_dir: Path, config: Path, wandb: str, debug: bool):
    params = Parameters(in_dir, out_dir, config, debug)
    setup_logging(params)
    setup_wandb(wandb, params, group='debug' if debug else 'eval')
    evaluate(params)
    wdb.finish()


def setup_wandb(params: Parameters, api_key: str, group: str = ''):
    wdb.login(key=api_key)
    wdb.init(project='FastIMRI-ReconAI', group=group, config=params.as_dict())


def setup_logging(params: Parameters):
    logging.basicConfig(
        filename=(params.out_dir / f'{params.name}.log').as_posix(),
        level=logging.DEBUG if params.debug else logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s >> %(message)s',
        datefmt='%H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-8s >> %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info(f"v{__version__}")
    logging.info(f'{params.name}\n{params}')
    logging.info(f"loading data from {params.in_dir.resolve()}\n")


if __name__ == '__main__':
    cli()
