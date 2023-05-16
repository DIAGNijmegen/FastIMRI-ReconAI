import logging

from .model.train import run_and_print_full_test
from os import listdir
from os.path import isfile, join
from .model.model_pytorch import CRNNMRI
import torch
from .parameters import Parameters
from .data.sequencebuilder import SequenceBuilder
from .data.dataloader import DataLoader
from .data.batcher import Batcher

def evaluation(params: Parameters):
    path = params.out_dir
    model_checkpoints = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'npz']

    for checkpoint in model_checkpoints:
        logging.info(f'starting {checkpoint}')
        params.config.data.sequence_length = int(checkpoint.split('.')[0].split('_')[2][3:])
        params.config.train.undersampling = int(checkpoint.split('.')[0].split('_')[1][0:2])

        logging.info('Started creating test batchers')
        dl_test = DataLoader(params.in_dir / 'test')
        dl_test.load(split_regex=params.config.data.split_regex, filter_regex=params.config.data.filter_regex)
        sequencer_test = SequenceBuilder(dl_test)

        kwargs = {'seed': params.config.data.sequence_seed,
                  'seq_len': params.config.data.sequence_length,
                  'mean_slices_per_mha': params.config.data.mean_slices_per_mha,
                  'max_slices_per_mha': params.config.data.max_slices_per_mha,
                  'q': params.config.data.q}
        test_sequences = sequencer_test.generate_multislice_sequences(**kwargs)
        test_batcher_equal = Batcher(dl_test)
        test_batcher_non_equal = Batcher(dl_test)

        for s in test_sequences.items():
            test_batcher_equal.append_sequence(sequence=s,
                                               crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x),
                                               norm=params.config.data.normalize,
                                               equal_images=True)
            test_batcher_non_equal.append_sequence(sequence=s,
                                                   crop_expand_to=(
                                                   params.config.data.shape_y, params.config.data.shape_x),
                                                   norm=params.config.data.normalize,
                                                   equal_images=False)
        logging.info('Finished creating test batchers')

        network = CRNNMRI(n_ch=1, nc=5).cuda()
        state_dict = torch.load(path / checkpoint)
        network.load_state_dict(state_dict)
        network.eval()

        name = checkpoint.split('.')[0]
        dirname = path / name
        dirname.mkdir(parents=True, exist_ok=True)

        logging.info('model loaded. Start eval')
        run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, dirname, name, 0, 0, '')
        # exit(1)
