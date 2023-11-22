import shutil
import json
from pathlib import Path

from click.testing import CliRunner
from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_segmentation, reconai_test_segmentation
from reconai.math.houghline import hough_line_prediction
import SimpleITK as sitk
import numpy as np

runner = CliRunner()


def test_train_segmentation():
    prepare_output_dir()
    run_click(reconai_train_segmentation, '--debug',
              in_dir='./tests/input/images',
              annotation_dir='./tests/input/annotations',
              out_dir='./tests/output/',
              sync_dir='./tests/output/rsync/')


def test_train_segmentation_existing():
    prepare_output_dir('nnUNet_raw', 'nnUNet_preprocessed')
    run_click(reconai_train_segmentation, '--debug', in_dir='./tests/output')

    fold_0 = Path(r'nnUNet_results\Dataset111_FastIMRI\nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d\fold_0')
    for pth in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        shutil.move('./tests/output' / fold_0 / pth, './tests/output_expected' / fold_0 / pth)


def test_test_segmentation():
    assert Path(r'./tests/output_expected/nnUNet_results/Dataset111_FastIMRI'
                r'/nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d/fold_0/checkpoint_best.pth').exists(), (
        FileNotFoundError('run ./tests/test_segmentation.py/test_train_segmentation_existing() to fix'))

    prepare_output_dir('nnUNet_results')
    run_click(reconai_test_segmentation,
              in_dir='./tests/input/images',
              nnunet_dir='./tests/output/',
              out_dir='./tests/output/nnUNet_predictions')


def test_hough_line_transform():
    annotations_dir = Path('./tests/input/annotations')
    output_dir = Path('./tests/output/houghlines')
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for file in annotations_dir.iterdir():
        if file.suffix == '.mha':
            annotation = sitk.GetArrayFromImage(sitk.ReadImage(file.as_posix()))
            with open(file.with_suffix('.json'), 'r') as f:
                facts = json.load(f)

            target_gnd = np.array(facts['inner_index'][:2])
            angle_gnd = facts['angle']
            target_pred, angle_pred = hough_line_prediction(annotation, show=output_dir / file.with_suffix('.png'))

            if target_pred is None:
                continue

            target_error = np.linalg.norm((target_gnd - target_pred).astype(np.float32))
            angle_error = np.abs(angle_gnd) - np.abs(angle_pred)

            results.append((target_error, angle_error))

