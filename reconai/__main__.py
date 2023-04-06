from pathlib import Path
import logging
from datetime import datetime
from typing import Dict

import click
from box import Box
from os.path import join

from .crnn_mri import train_network, test_accelerations


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--debug', is_flag=True, default=False, help="light weight process for debugging")
@click.option('--in_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--sequence_len', type=int, default=15, help='number of frames')
@click.option('-f', '--folds', count=True, help='number of folds, -fff for three folds')
@click.option('--num_epoch', type=int, default=250, help='number of epochs')
@click.option('--loss', type=click.Choice(['mse', 'mse+ssim', 'ssim'], case_sensitive=False),
              default='mse', help='loss function')
@click.option('--batch_size', type=int, default=1, help='batch size')
@click.option('--lr', type=float, default=0.001, help='initial learning rate')
@click.option('--acceleration_factor', type=float, default=8.0, help='acceleration factor for k-space sampling')
@click.option('--complex', is_flag=True, type=bool, default=False)
@click.option('--seed', type=int, default=None, help='train/test shuffling seed')
@click.option('--test_accelerations', is_flag=True, type=bool, default=False)
def train_recon(**kwargs):
    setup_logging('train_recon', kwargs, test=kwargs['debug'])

    try:
        if kwargs['test_accelerations']:
            test_accelerations(Box(kwargs))
        else:
            train_network(Box(kwargs))
    except Exception as e:
        logging.exception(e)


def setup_logging(name: str, kwargs: Dict, root_dir: Path = None, test: bool = False):
    kwargs['date'] = datetime.now().strftime("%Y%m%d_%H%M")
    if not root_dir:
        root_dir: Path = kwargs['out_dir']
    logging.basicConfig(
        filename=join(root_dir.as_posix(), f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'),
        level=logging.DEBUG if test else logging.INFO,
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
    # logging.info(f"v{__version__}\n")
    [logging.info(f'{key}: {value}') for key, value in kwargs.items()]
    logging.info(f"loading data from {kwargs['in_dir'].absolute()}\n")


if __name__ == '__main__':
    cli()
