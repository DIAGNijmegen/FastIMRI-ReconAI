import shutil
from pathlib import Path

from click.testing import CliRunner

from reconai.__main__ import reconai_train_segmentation

runner = CliRunner()


def test_segmentation():
    for directory in ['./tests/output/nnUNet_raw', './tests/output/nnUNet_results']:
        if (directory := Path(directory)).exists():
            shutil.rmtree(directory)

    kwargs = {
        'in_dir': './tests/input/images',
        'annotation_dir': './tests/input/annotations',
        'out_dir': './tests/output/'
    }

    args = []
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)

    result = runner.invoke(reconai_train_segmentation, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0


def test_existing_segmentation():
    for directory in ['./tests/output/nnUNet_raw', './tests/output/nnUNet_results']:
        if (directory := Path(directory)).exists():
            shutil.rmtree(directory)
    shutil.copytree('./tests/output_expected/nnUNet_raw', './tests/output/nnUNet_raw')

    result = runner.invoke(reconai_train_segmentation, ['--in_dir', './tests/output/nnUNet_raw'])
    if result.exception:
        raise result.exception
    assert result.exit_code == 0