import json
import os
from pathlib import Path

import SimpleITK as sitk
import matplotlib
import numpy as np
from click.testing import CliRunner
import pytest

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from conftest import run_click, prepare_output_dir
from reconai.__main__ import reconai_test
from reconai.evaluation import predict, Prediction

runner = CliRunner()

crnns = ['20231113T1603_CRNN-MRI_R2_E3_DEBUG_1', '20230830T1030_CRNN-MRI_R2_E3_DEBUG']


def test_test():
    prepare_output_dir(*crnns)
    for crnn in crnns:
        run_click(reconai_test, '--debug',
                  in_dir='./tests/input/images',
                  model_dir=f'./tests/output/{crnn}')


@pytest.mark.parametrize("inference_information", [False, True])
def test_test_nnunet(inference_information):
    prepare_output_dir(*crnns, 'nnUNet_preprocessed', 'nnUNet_results')
    if not inference_information:
        os.remove(f'./tests/output/nnUNet_results/Dataset111_FastIMRI/inference_information.json')
    for crnn in crnns:
        run_click(reconai_test, '--debug',
                  in_dir='./tests/input/images',
                  model_dir=f'./tests/output/{crnn}',
                  nnunet_dir='./tests/output',
                  annotations_dir='./tests/input/annotations')


def test_predict():
    prepare_output_dir()

    output_dir = Path('./tests/output')
    output_dir.mkdir(exist_ok=True)

    prediction_strategies = Prediction.STRATEGIES
    results = {strat: [] for strat in prediction_strategies}
    for i, file in enumerate(Path('./tests/input/annotations').iterdir()):
        if file.suffix == '.mha':
            annotation = sitk.GetArrayFromImage(sitk.ReadImage(file.as_posix()))
            with open(file.with_suffix('.json'), 'r') as f:
                facts = json.load(f)

            target_gnd: tuple[int, int] = facts['inner'][:2]
            angle_gnd: float = facts['angle']
            for strategy in prediction_strategies:
                pred = predict(annotation[2], (target_gnd[0], target_gnd[1], angle_gnd), strategy=strategy)
                pred.save(output_dir / (f'{i}_{strategy}_' + file.with_suffix('.png').name), debug=True)
                results[strategy].append(pred.error())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    bar_width = 0.3
    indices = np.arange(len(results[prediction_strategies[0]]))

    for i, label in enumerate(['Error in mm', 'Error in degrees']):
        for j, (strategy, errors) in enumerate(results.items()):
            axes[i].bar(indices - bar_width * (len(prediction_strategies) - 1) / 2 + j * bar_width, [e[i] for e in errors], bar_width, label=strategy)

        axes[i].set_xlabel('Annotation')
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} per Strategy')
        axes[i].set_xticks(indices)
        axes[i].legend()

    # Displaying the plot
    plt.tight_layout()
    fig.savefig((output_dir / 'results.png').as_posix())

