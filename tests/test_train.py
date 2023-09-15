import shutil
import json
from pathlib import Path

import pandas
import freezegun
from click.testing import CliRunner

from reconai import train_recon

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_debug(monkeypatch):
    out_dir = Path('./tests/output/20230830-1030_CRNN-MRI_R8_E5_DEBUG')
    if out_dir.exists():
        shutil.rmtree(out_dir)

    secrets_path = Path('./tests/input/secrets.json')
    secrets = dict()
    if secrets_path.exists():
        with open(secrets_path, 'r') as j:
            secrets = json.load(j)

    kwargs = {
        'in_dir': './tests/input/data',
        'out_dir': out_dir.parent.as_posix(),
        'wandb': secrets.get('wandb', ''),
        'debug': None
    }
    args = []
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)

    result = runner.invoke(train_recon, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
