import shutil
from pathlib import Path

from click.testing import CliRunner

from reconai.__main__ import reconai_train_segmentation, reconai_test_segmentation

runner = CliRunner()


def test_train_segmentation():
    raw, preprocessed, results = 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'
    for name in [raw, preprocessed, results]:
        if (directory := Path(f'./tests/output/{name}')).exists():
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

    result = runner.invoke(reconai_train_segmentation, args + ['--debug'])
    if result.exception:
        raise result.exception
    assert result.exit_code == 0


def test_train_segmentation_existing():
    raw, preprocessed, results = 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'
    for name in [raw, preprocessed, results]:
        if (directory := Path(f'./tests/output/{name}')).exists():
            shutil.rmtree(directory)
        shutil.copytree(f'./tests/output_expected/{name}', f'./tests/output/{name}')

    result = runner.invoke(reconai_train_segmentation, ['--in_dir', './tests/output/nnUNet_raw', '--debug'])
    if result.exception:
        raise result.exception
    assert result.exit_code == 0

    fold_0 = Path(r'nnUNet_results\Dataset111_FastIMRI\nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d\fold_0')
    for pth in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        shutil.move('./tests/output' / fold_0 / pth, './tests/output_expected' / fold_0 / pth)


def test_test_segmentation():
    raw, preprocessed, results = 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'
    for name in [raw, preprocessed, results]:
        if (directory := Path(f'./tests/output/{name}')).exists():
            shutil.rmtree(directory)
        shutil.copytree(f'./tests/output_expected/{name}', f'./tests/output/{name}')