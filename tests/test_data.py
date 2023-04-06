from pathlib import Path

from reconai.data.data import gather_data
from reconai.data.Batcher import Batcher


def test_volume():
    data = gather_data(Path('input'))
    data_error = Batcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    # data_n = len(data)

    batcher = Batcher(data)
    for item in batcher.generate():
        # TODO: check this with input yaml config file rather than constant val
        assert item.shape == (1, 15, 256, 256)
    pass
