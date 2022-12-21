from pathlib import Path

import click
from box import Box

from .crnn_mri import crnn_mri


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('--test', is_flag=True, default=False, help="light weight version for testing")
@click.option('--data_dir', type=Path, required=True)
@click.option('--out_dir', type=Path, required=True)
@click.option('--T', type=int, default=15, help='number of frames')
@click.option('--num_epoch', type=int, default=10, help='number of epochs')
@click.option('--batch_size', type=int, default=1, help='batch size')
@click.option('--lr', type=float, default=0.001, help='initial learning rate')
@click.option('--acceleration_factor', type=float, default=8.0, help='acceleration factor for k-space sampling')
@click.option('--complex', is_flag=True)
@click.option('--savefig', is_flag=True)
def train(**kwargs):
    [print(f'{key}: {value}') for key, value in kwargs.items()]
    crnn_mri(Box(kwargs))


if __name__ == '__main__':
    cli()