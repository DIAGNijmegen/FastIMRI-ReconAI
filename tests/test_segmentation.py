import shutil
import json
from pathlib import Path

from click.testing import CliRunner
from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_train_segmentation, reconai_test_segmentation
from reconai.predict import predict_target
import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


def test_test_segmentation():
    assert Path(r'./tests/output_expected/nnUNet_results/Dataset111_FastIMRI'
                r'/nnUNetTrainer_FastIMRI_debug__nnUNetPlans__2d/fold_0/checkpoint_best.pth').exists(), (
        FileNotFoundError('run ./tests/test_segmentation.py/test_train_segmentation_existing() to fix'))

    prepare_output_dir('nnUNet_results')
    run_click(reconai_test_segmentation,
              in_dir='./tests/input/images',
              nnunet_dir='./tests/output/',
              out_dir='./tests/output/nnUNet_predictions')


def test_predict():
    prepare_output_dir()

    output_dir = Path('./tests/output')
    output_dir.mkdir(exist_ok=True)

    strategies = ['pca', 'hough_line_transform']
    results = {strat: [] for strat in strategies}
    for i, file in enumerate(Path('./tests/input/annotations').iterdir()):
        if file.suffix == '.mha':
            annotation = sitk.GetArrayFromImage(sitk.ReadImage(file.as_posix()))
            with open(file.with_suffix('.json'), 'r') as f:
                facts = json.load(f)

            target_gnd = np.array(facts['inner'][:2])
            angle_gnd = facts['angle']
            for strategy in strategies:
                pred = predict_target(annotation, strategy=strategy)
                if pred:  # gnd_angle is not correct wrt image space
                    pred.set_ground_truth(*target_gnd, angle_gnd)
                    pred.show(save=output_dir / (f'{i}_{strategy}_' + file.with_suffix('.png').name))

                    target_error = np.linalg.norm((target_gnd - pred.target).astype(np.float32))
                    angle_error = np.abs(angle_gnd - pred.angle)
                    results[strategy].append((target_error, np.rad2deg(angle_error)))
                else:
                    results[strategy].append((-1, -1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    bar_width = 0.3
    indices = np.arange(len(results[strategies[0]]))

    for i, label in enumerate(['Error in mm', 'Error in degrees']):
        for j, (strategy, errors) in enumerate(results.items()):
            axes[i].bar(indices - bar_width * (len(strategies) - 1) / 2 + j * bar_width, [e[i] for e in errors], bar_width, label=strategy)

        axes[i].set_xlabel('Annotation')
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} per Strategy')
        axes[i].set_xticks(indices)
        axes[i].legend()

    # Displaying the plot
    plt.tight_layout()
    fig.savefig((output_dir / 'results.png').as_posix())

