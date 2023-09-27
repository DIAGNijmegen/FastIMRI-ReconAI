import json
import shutil
from pathlib import Path

import freezegun
from click.testing import CliRunner

from conftest import run_click
from reconai.__main__ import reconai_train_reconstruction, reconai_test_reconstruction

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_reconstruction(monkeypatch):
    output_dir = Path('./tests/output')
    for d in output_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    secrets_path = Path('./tests/input/secrets.json')
    if not secrets_path.exists():
        raise FileNotFoundError(f'no secrets.json file found at {secrets_path}')

    with open(secrets_path, 'r') as j:
        secrets = json.load(j)

    run_click(reconai_train_reconstruction,
              in_dir='./tests/input/images',
              out_dir=output_dir.as_posix(),
              wandb_api=secrets['wandb'])


def test_test_reconstruction():
    model_dir = Path('./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG')
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(Path('./tests/output_expected/20230830T1030_CRNN-MRI_R2_E3_DEBUG'), model_dir)

    run_click(reconai_test_reconstruction, in_dir='./tests/input/images', model_dir=model_dir.as_posix())
