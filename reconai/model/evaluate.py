#!/usr/bin/env python
import logging
import time
from .train import run_and_print_full_test
from os import listdir
from os.path import isfile, join
from .model_pytorch import CRNNMRI
import torch
from reconai.parameters import Parameters
from reconai.data.sequencebuilder import SequenceBuilder
from reconai.data.dataloader import DataLoader
from reconai.data.batcher import Batcher
from reconai.data.data import prepare_input_as_variable
import numpy as np

def evaluate(params: Parameters):
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
        # test_batcher_non_equal = Batcher(dl_test)

        # for s in test_sequences.items():
        #     test_batcher_equal.append_sequence(sequence=s,
        #                                        crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x),
        #                                        norm=params.config.data.normalize,
        #                                        equal_images=True)
        #     test_batcher_non_equal.append_sequence(sequence=s,
        #                                            crop_expand_to=(
        #                                             params.config.data.shape_y,
        #                                             params.config.data.shape_x),
        #                                            norm=params.config.data.normalize,
        #                                            equal_images=False)
        seq = next(test_sequences.items())
        test_batcher_equal.append_sequence(sequence=seq,
                                           crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x),
                                           norm=params.config.data.normalize,
                                           equal_images=True)
        logging.info('Finished creating test batchers')

        network = CRNNMRI(n_ch=1, nf=64, ks=3, nc=5, nd=5).cuda()
        state_dict = torch.load(path / checkpoint)
        network.load_state_dict(state_dict)
        network.eval()

        name = checkpoint.split('.')[0]
        dirname = path / name
        dirname.mkdir(parents=True, exist_ok=True)

        logging.info('model loaded. Start eval')
        times = []
        for i in range(3):
            with torch.no_grad():
                img = next(test_batcher_equal.items())
                im_und, k_und, mask, im_gnd = prepare_input_as_variable(img,
                                                                        11,
                                                                        params.config.train.undersampling,
                                                                        equal_mask=True)
                t_start = time.time()
                _, _ = network(im_und, k_und, mask, test=False)
                t_end = time.time()
                times.append(t_end - t_start)
                logging.info(f'actual inference speed; total time: {t_end - t_start}')
        logging.info(f'average actual inference speed {np.average(times)}')

        # run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, dirname, name, 0, 0, 0, '')
        # exit(1)
