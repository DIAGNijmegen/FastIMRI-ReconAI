import json
from pathlib import Path

from click.testing import CliRunner

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


def test_test_reconstruction():
    prepare_output_dir('20230830T1030_CRNN-MRI_R2_E3_DEBUG')
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir='./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG')


def test_test_reconstruction_nnunet():
    prepare_output_dir('20230830T1030_CRNN-MRI_R2_E3_DEBUG', 'nnUNet_results')
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir='./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG',
              nnunet_dir='./tests/output',
              annotations_dir='./tests/input/annotations')


def test_test_reconstruction_nnunet_with_annotations():
    prepare_output_dir('20230830T1030_CRNN-MRI_R2_E3_DEBUG', 'nnUNet_results')
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir='./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG',
              nnunet_dir='./tests/output',
              annotations_dir='./tests/input/annotations')
