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


@pytest.mark.parametrize('model', ['Vanilla', 'Zealot'])
@pytest.mark.parametrize('und', [8, 16])
@pytest.mark.parametrize('data', ['realtime', 'slice', 'example'])
@pytest.mark.parametrize('preprocessing', ['none', 'abs', 'simulated'])
def test_fire_module(output_dir, model: str, und: int, data: str, preprocessing: str):
    if data == 'example' and preprocessing != 'simulated':
        shutil.rmtree(output_dir)
        pytest.skip('meaningless combination of parameters')

    input_dir = Path(f'tests/input/model_{model}_{und}')
    if not input_dir.exists():
        shutil.rmtree(output_dir)
        pytest.skip(f'{model}_{und} does not exist')

    experiment = {'model': model, 'und': und, 'data': data, 'preprocessing': preprocessing}
    array = np.load(Path(input_dir / f'../data/{data}.npy'))

    logger = logging.getLogger('test_fire_module')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    module = FireReconstruct()
    module.logger = logger
    module.load(model_dir=input_dir, debug=True, experiment=experiment)
    for _ in module.run(array, {}):
        module.export(output_dir)
