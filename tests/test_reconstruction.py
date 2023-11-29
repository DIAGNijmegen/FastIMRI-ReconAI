import json
from pathlib import Path

from click.testing import CliRunner

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_reconstruction

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
