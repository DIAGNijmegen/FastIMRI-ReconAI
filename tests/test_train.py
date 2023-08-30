import pytest
import pytest_click
import torch
import numpy as np

import click
from datetime import datetime

import freezegun
from click.testing import CliRunner

from reconai import train_recon

runner = CliRunner()


@freezegun.freeze_time("2023-08-30 10:30:00")
def test_train_debug(monkeypatch):
    kwargs = {
        'in_dir': './input',
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
        print(result.exception)
    assert result.exit_code == 0
    # assert "Expected Output" in result.output

# # Add more test cases for other Click commands
#
#
# # from reconai import
# from reconai.model.kspace_pytorch import DataConsistencyInKspace
# from reconai.data.data import prepare_input
#
# @pytest_click.cli_runner
# def test_train(cli):
#     cli(debug=True)
#
#
# @pytest.mark.usefixtures("batcher")
# def test_train(batcher):
#     for im in batcher.items():
#         im_und, k_und, mask, im_gnd = prepare_input(im, 1, 1)
#
#         pass
