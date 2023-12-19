from pathlib import Path

from reconai.parameters import ModelTrainParameters


def test_parameters():
    in_dir, out_dir = Path('./input'), Path('./output')
    p = ModelTrainParameters(in_dir, out_dir)
    assert p.data.shape_x == 256
    p = ModelTrainParameters(in_dir, out_dir, 'data:\n shape_x: 56')
    assert p.data.shape_x == 56
    assert p.as_dict()['data']['shape_x'] == 56
