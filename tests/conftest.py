import shutil
from typing import Callable
from pathlib import Path

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


def prepare_output_dir(*dirnames_to_remove: str, replace_with_expected: bool = False):
    for name in dirnames_to_remove:
        if (directory := Path(f'./tests/output/{name}')).exists():
            shutil.rmtree(directory)
        if replace_with_expected and (directory_expected := Path(f'./tests/output_expected/{name}')).exists():
            shutil.copytree(directory_expected, f'./tests/output/{name}')
