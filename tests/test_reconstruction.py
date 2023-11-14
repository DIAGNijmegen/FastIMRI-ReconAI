import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_reconstruction, reconai_test_reconstruction

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


@pytest.mark.parametrize('single', [True, False])
def test_test_reconstruction(single: bool):
    crnn = '20231113T1603_CRNN-MRI_R2_E3_DEBUG_1' if single else '20230830T1030_CRNN-MRI_R2_E3_DEBUG'
    prepare_output_dir(crnn)
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir=f'./tests/output/{crnn}')


@pytest.mark.parametrize('single', [True])
def test_test_reconstruction_nnunet(single: bool):
    crnn = '20231113T1603_CRNN-MRI_R2_E3_DEBUG_1' if single else '20230830T1030_CRNN-MRI_R2_E3_DEBUG'
    prepare_output_dir(crnn, 'nnUNet_results')
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir=f'./tests/output/{crnn}',
              nnunet_dir='./tests/output',
              annotations_dir='./tests/input/annotations')
