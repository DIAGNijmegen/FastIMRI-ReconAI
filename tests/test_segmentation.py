import shutil
from pathlib import Path

import matplotlib
from click.testing import CliRunner

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_segmentation

matplotlib.use('TkAgg')

runner = CliRunner()


def test_train_segmentation():
    prepare_output_dir()
    run_click(reconai_train_segmentation, '--debug',
              in_dir='./tests/input/images',
              annotations_dir='./tests/input/annotations',
              out_dir='./tests/output/',
              sync_dir='./tests/output/rsync/')


def test_train_segmentation_existing():
    prepare_output_dir('nnUNet_raw', 'nnUNet_preprocessed')
    run_click(reconai_train_segmentation, '--debug', in_dir='./tests/output')

    fold_0 = Path(r'nnUNet_results\Dataset111_FastIMRI\nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d\fold_0')
    for pth in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        shutil.move('./tests/output' / fold_0 / pth, './tests/output_expected' / fold_0 / pth)
