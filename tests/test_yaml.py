from pathlib import Path
from importlib import resources

from reconai.config import load, load_str


def test_config():
    load(Path(str(resources.path('reconai.resources', 'config_default.yaml'))))


def test_arbitrary_valid_yaml():
    load_str('')
    load_str('experiment:')
    load_str('data:\n shape_x: 256')