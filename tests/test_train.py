import shutil

import freezegun
from click.testing import CliRunner

from reconai import train_recon

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_debug(monkeypatch):
    shutil.rmtree('./output/20230830-1030_CRNN-MRI_R8_E5_DEBUG')

    kwargs = {
        'in_dir': './input/patient1',
        'out_dir': './output',
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
