import shutil
from pathlib import Path

import freezegun
from click.testing import CliRunner

from reconai import train_recon

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_debug(monkeypatch):
    out_dir = Path('./output/20230830-1030_CRNN-MRI_R8_E5_DEBUG')
    if out_dir.exists():
        shutil.rmtree(out_dir)

    kwargs = {
        'in_dir': './tests/input',
        'out_dir': out_dir.parent.as_posix(),
        'debug': None
    }
    args = []
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)
    # keep date same
    result = runner.invoke(train_recon, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
