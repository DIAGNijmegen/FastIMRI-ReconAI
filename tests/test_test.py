import shutil
import json
from pathlib import Path

import freezegun
from click.testing import CliRunner

from reconai.__main__ import reconai_test

runner = CliRunner()


def test_test_debug():
    model_dir = Path('./tests/output_expected/20230830T1030_CRNN-MRI_R2_E3_DEBUG')

    secrets_path = Path('./tests/input/secrets.json')
    if not secrets_path.exists():
        raise FileNotFoundError(f'no secrets.json file found at {secrets_path}')

    with open(secrets_path, 'r') as j:
        secrets = json.load(j)

    kwargs = {
        'in_dir': './tests/input/data',
        'model_dir': model_dir.as_posix(),
        'wandb_api': secrets['wandb']
    }

    args = []
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)

    result = runner.invoke(reconai_test, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
