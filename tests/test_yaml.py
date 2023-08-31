from pathlib import Path
from importlib import resources

from reconai.parameters import Parameters


def test_parameters():
    in_dir, out_dir = './input', './output'
    p = Parameters(in_dir, out_dir)
    assert p.data.shape_x == 256
    p = Parameters(in_dir, out_dir, 'data:\n shape_x: 56')
    assert p.data.shape_x == 56
