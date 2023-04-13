from pathlib import Path
import logging
from typing import List

import click
from os.path import join

from .parameters import Parameters
from .models.bcrnn import __package__ as bcrnn, train as bcrnn_train
from .models.kiki import __package__ as kiki, train as kiki_train
from .__version__ import __version__


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--model', type=click.Choice(Parameters.model_names(bcrnn, kiki), case_sensitive=False), required=True)
@click.option('--config', type=Path, required=False)
@click.option('--debug', is_flag=True, default=False, help="light weight process for debugging")
def train_recon(in_dir: Path, out_dir: Path, model: str, config: Path, debug: bool):
    params = Parameters(in_dir, out_dir, model, config, debug)

    setup_logging(params)

    if params.model == bcrnn:
        bcrnn_train(params)
    elif params.model == kiki:
        kiki_train(params)
    raise ValueError(f'unknown --model: {model}')

    # try:
    #     if kwargs['test_accelerations']:
    #         test_accelerations(Box(kwargs))
    #     else:
    #         train_network(Box(kwargs))
    # except Exception as e:
    #     logging.exception(e)


def setup_logging(params: Parameters):
    logging.basicConfig(
        filename=join(params.out_dir.as_posix(), f'{params.name_date}.log'),
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
