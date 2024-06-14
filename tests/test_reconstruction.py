import json
import logging
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import imageio

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_reconstruction, reconai_reconstruct
from reconai.fire import FireReconstruct

runner = CliRunner()


def test_train_reconstruction():
    secrets_path = Path('./tests/input/secrets.json')
    if not secrets_path.exists():
        raise FileNotFoundError(f'no secrets.json file found at {secrets_path}')
    with open(secrets_path, 'r') as j:
        secrets = json.load(j)

    prepare_output_dir()
    run_click(reconai_train_reconstruction,
              in_dir='./tests/input/images',
              out_dir='./tests/output',
              wandb_api=secrets['wandb'])


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


def test_fire_module():
    output_dir, = prepare_output_dir('fire_module')

    logger = logging.getLogger('test_fire_module')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    module = FireReconstruct()
    module.logger = logger
    module.load(model_dir='tests/input/realtime/', debug=True)
    array = np.load(Path('tests/input/realtime/IMRI#SRDMR5#F31277#M212#D280823#T150233#imri_trufitrans.npy'))
    for _ in module.run(array, {}):
        module.export(Path(output_dir))

