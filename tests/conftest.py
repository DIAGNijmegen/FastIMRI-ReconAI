# noinspection PyUnresolvedReferences

# Use this file to share fixtures across multiple files.
# They can be accessed via @pytest.mark.usefixtures("<name of fixture>")

# flake8: noqa
from .test_data import batcher, sequences, dataloader
from reconai.data.sequencebuilder import SequenceBuilder

SequenceBuilder.MAX_WORKERS = 1
