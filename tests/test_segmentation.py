import shutil
from pathlib import Path

from click.testing import CliRunner

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_segmentation, reconai_test_segmentation

runner = CliRunner()


def test_train_segmentation():
    prepare_output_dir('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results', replace_with_expected=False)
    run_click(reconai_train_segmentation, '--debug',
              in_dir='./tests/input/images',
              annotation_dir='./tests/input/annotations',
              out_dir='./tests/output/')


def test_train_segmentation_existing():
    prepare_output_dir('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results', replace_with_expected=True)
    run_click(reconai_train_segmentation, '--debug', in_dir='./tests/output/nnUNet_raw')

    fold_0 = Path(r'nnUNet_results\Dataset111_FastIMRI\nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d\fold_0')
    for pth in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        shutil.move('./tests/output' / fold_0 / pth, './tests/output_expected' / fold_0 / pth)


def test_test_segmentation():
    assert Path(r'./tests/output_expected/nnUNet_results/Dataset111_FastIMRI'
                r'/nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d/fold_0/checkpoint_best.pth').exists(), (
        FileNotFoundError('run ./tests/test_segmentation.py/test_train_segmentation_existing() to fix'))

    prepare_output_dir('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results', 'nnUNet_predictions',
                       replace_with_expected=True)
    run_click(reconai_test_segmentation, '--debug',
              in_dir='./tests/input/images',
              nnunet_dir='./tests/output/nnUNet_results',
              out_dir='./tests/output/nnUNet_predictions')
