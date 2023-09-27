import json
import shutil
from pathlib import Path

import freezegun
from click.testing import CliRunner

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_reconstruction, reconai_test_reconstruction

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_reconstruction(monkeypatch):
    prepare_output_dir('20230830T1030_CRNN-MRI_R2_E3_DEBUG')

    secrets_path = Path('./tests/input/secrets.json')
    if not secrets_path.exists():
        raise FileNotFoundError(f'no secrets.json file found at {secrets_path}')
    with open(secrets_path, 'r') as j:
        secrets = json.load(j)

    run_click(reconai_train_reconstruction,
              in_dir='./tests/input/images',
              out_dir='./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG',
              wandb_api=secrets['wandb'])


def test_test_reconstruction():
    prepare_output_dir('20230830T1030_CRNN-MRI_R2_E3_DEBUG', replace_with_expected=True)
    run_click(reconai_test_reconstruction,
              in_dir='./tests/input/images',
              model_dir='./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG')
