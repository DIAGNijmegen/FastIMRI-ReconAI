from pathlib import Path

from reconai.yaml import load, load_str


def test_config():
    load(Path('test_config.yaml'))


def test_arbitrary_yaml():
    load_str('')
    load_str('experiment:')
    load_str('volume:\n shape:')