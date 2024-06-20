import os
import shutil
from typing import Callable
from pathlib import Path

import pytest
from click import BaseCommand
from click.testing import CliRunner

runner = CliRunner()


def run_click(func: Callable | BaseCommand, *args, **kwargs):
    args = list(args)
    for key, value in kwargs.items():
        args.append(f'--{key}')
        if value:
            args.append(value)

    result = runner.invoke(func, args)
    if result.exception:
        raise result.exception
    assert result.exit_code == 0


@pytest.fixture(scope='function')
def output_dir(request):
    output_dir = Path(f'./tests/output/{request.node.name}')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    yield output_dir


def prepare_output_dir(*dirnames_to_copy: str) -> list[Path]:
    for path in Path('./tests/output').iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)

    outputs = []
    for path_expected in Path('./tests/output_expected').iterdir():
        if path_expected.name in dirnames_to_copy:
            outputs.append(Path(f'./tests/output/{path_expected.name}'))
            shutil.copytree(path_expected, outputs[-1])

    return list(outputs)
