import json
import logging
import shutil
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pytest
import torch

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_reconstruction, reconai_reconstruct
from reconai.fire import FireReconstruct

runner = CliRunner()


def test_train_reconstruction():
    secrets_path = Path('./tests/input/secret.json')
    kwargs = {'in_dir': './tests/input/images', 'out_dir': './tests/output'}
    if secrets_path.exists():
        with open(secrets_path, 'r') as j:
            kwargs['wandb_api'] = json.load(j)['wandb']

    prepare_output_dir()
    run_click(reconai_train_reconstruction, **kwargs)


def test_train_reconstruction_320():
    kwargs = {'in_dir': './tests/input/images', 'out_dir': './tests/output', 'config': './tests/input/test_train_reconstruction_320.yaml'}
    prepare_output_dir()
    run_click(reconai_train_reconstruction, **kwargs)


def test_reconstruct():
    model = '20230830T1030_CRNN-MRI_R2_E3_DEBUG'
    prepare_output_dir(model)
    run_click(reconai_reconstruct,
              in_dir='./tests/input/images',
              model_dir=f'./tests/output/{model}',
              out_dir='./tests/output')


def test_reconstruct_kspace():
    model = '20230830T1030_CRNN-MRI_R2_E3_DEBUG'
    prepare_output_dir(model)
    run_click(reconai_reconstruct,
              in_dir='./tests/input/kspace',
              model_dir=f'./tests/output/{model}',
              out_dir='./tests/output',
              out_png=True)


def test_reconstruct_r8():
    prepare_output_dir('r8')
    run_click(reconai_reconstruct,
              in_dir='./tests/input/images',
              model_dir=f'./tests/input/realtime',
              out_dir='./tests/output/r8',
              out_png=True)


experiments = ([['example', '16'], ['example', 'realtime'], ['example', '16', 'realtime'], ['simulated', '16'],
               ['simulated', 'realtime'], ['simulated', '16', 'realtime'], ['abs', '16'],
               ['abs', 'realtime'], ['abs', '16', 'realtime'], ['16'], ['realtime'], ['16', 'realtime'],
                ['example'], ['simulated'], ['abs'], []])


@pytest.mark.parametrize("experiment", experiments)
def test_fire_module(output_dir, experiment):
    output_dir_renamed = output_dir.parent / (output_dir.stem + '[' + '_'.join(sorted(experiment)) + ']')
    if output_dir_renamed.exists():
        shutil.rmtree(output_dir_renamed)
    output_dir = output_dir.rename(output_dir_renamed)
    model_name = 'Zealot'

    if '16' in experiment:
        return
        input_dir = f'tests/input/model_{model_name}_16/'
    else:
        input_dir = f'tests/input/model_{model_name}/'

    if 'realtime' in experiment:
        array = np.load(Path(input_dir + '../data/realtime.npy'))
    else:
        array = np.load(Path(input_dir + '../data/slice.npy'))

    logger = logging.getLogger('test_fire_module')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    module = FireReconstruct()
    module.logger = logger
    module.load(model_dir=input_dir, debug=True, experiment=experiment)
    for _ in module.run(array, {}):
        module.export(output_dir)
