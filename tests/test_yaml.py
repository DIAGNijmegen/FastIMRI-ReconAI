from pathlib import Path
from importlib import resources

from reconai.parameters import load, load_str
from reconai.parameters import Parameters


def test_parameters():
    p = Parameters()
    p = Parameters('data:\n shape_x: 56')
    assert p.data.shape_x == 56


def test_config():
    load(Path(str(resources.path('reconai.resources', 'config_default.yaml'))))


def test_arbitrary_valid_yaml():
    load_str('')
    load_str('experiment:')
    load_str('data:\n shape_x: 256')