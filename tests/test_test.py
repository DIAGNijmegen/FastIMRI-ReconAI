import shutil
from pathlib import Path

from click.testing import CliRunner

from reconai.__main__ import reconai_test

runner = CliRunner()


def test_test():
    model_dir = Path('./tests/output/20230830T1030_CRNN-MRI_R2_E3_DEBUG')
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(Path('./tests/output_expected/20230830T1030_CRNN-MRI_R2_E3_DEBUG'), model_dir)

    kwargs = {
        'in_dir': './tests/input/images',
        'model_dir': model_dir.as_posix()
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
