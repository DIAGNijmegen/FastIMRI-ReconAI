from pathlib import Path
import logging

import click
import shutil
from os.path import join

from .parameters import Parameters
from .model import train
from .evaluation import evaluation
from .__version__ import __version__


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=False)
@click.option('--debug', is_flag=True, default=False, help="light weight process for debugging")
def train_recon(in_dir: Path, out_dir: Path, config: Path, debug: bool):
    params = Parameters(in_dir, out_dir, config, debug)
    save_dir: Path = params.out_dir / params.date_name
    save_dir.mkdir(parents=True)
    shutil.copy(config, save_dir / 'config.yaml')
    params.out_dir = save_dir
    setup_logging(params)
    train(params)


@cli.command(name='eval')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--config', type=Path, required=True)
def evaluate_models(in_dir: Path, out_dir: Path, config: Path):
    params = Parameters(in_dir, out_dir, config, False)
    setup_logging(params)
    evaluation(params)


def setup_logging(params: Parameters):
    logging.basicConfig(
        filename=join(params.out_dir.as_posix(), f'{params.date_name}.log'),
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
    logging.info(f'{params.name}\n{params.as_yaml()}')
    logging.info(f"loading data from {params.in_dir.resolve()}\n")


if __name__ == '__main__':
    cli()
