from typing import Callable

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
