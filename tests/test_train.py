import shutil
import json
from pathlib import Path

import pandas
import freezegun
from click.testing import CliRunner

from reconai.__main__ import reconai_train

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train(monkeypatch):
    output_dir = Path('./tests/output')
    for d in output_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    secrets_path = Path('./tests/input/secrets.json')
    if not secrets_path.exists():
        raise FileNotFoundError(f'no secrets.json file found at {secrets_path}')

    with open(secrets_path, 'r') as j:
        secrets = json.load(j)

    kwargs = {
        'in_dir': './tests/input/images',
        'out_dir': output_dir.as_posix(),
        'wandb_api': secrets['wandb']
    }
    args = []
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)

    result = runner.invoke(reconai_train, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
