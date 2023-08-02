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
from reconai.utils.graph import set_ax
import numpy as np
import matplotlib.pyplot as plt
from reconai.model.test import run_testset
from reconai.utils.metric import ssim
from scipy.stats import ttest_ind
from reconai.model.dnn_io import from_tensor_format

def evaluate(params: Parameters):
    path = params.out_dir
    model_checkpoints = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'npz']

    print_comparisons(params)
    exit(1)

    # sig()
    # exit(1)

    for checkpoint in model_checkpoints:
        logging.info(f'starting {checkpoint}')
        params.config.data.sequence_length = int(checkpoint.split('.')[0].split('_')[2][3:])
        params.config.train.undersampling = int(checkpoint.split('.')[0].split('_')[1][0:2])

        logging.info('Started creating test batchers')
        filter_regex = 'sag'
        dl_test = DataLoader(params.in_dir / 'test')
        dl_test.load(split_regex=params.config.data.split_regex, filter_regex=filter_regex)
        sequencer_test = SequenceBuilder(dl_test)

        kwargs = {'seed': params.config.data.sequence_seed,
                  'seq_len': params.config.data.sequence_length,
                  # 'mean_slices_per_mha': params.config.data.mean_slices_per_mha,
                  # 'max_slices_per_mha': params.config.data.max_slices_per_mha,
                  # 'q': params.config.data.q
                  }
        test_sequences = sequencer_test.generate_singleslice_sequences(**kwargs)
        logging.info(len(test_sequences))

        test_batcher = Batcher(dl_test)

        for s in test_sequences.items():
            # seq = next(test_sequences.items())
            test_batcher.append_sequence(sequence=s,
                                         crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x),
                                         norm=params.config.data.normalize,
                                         equal_images='nonequal' not in checkpoint)
        logging.info('Finished creating test batchers')

        network = CRNNMRI(n_ch=1, nf=64, ks=3, nc=5, nd=5, equal='nonequal' not in checkpoint).cuda()
        state_dict = torch.load(path / checkpoint)
        network.load_state_dict(state_dict)
        network.eval()

        vis_nenm, iters_nenm, base_psnr_nenm, test_psnr_nenm, test_batches_nenm = \
            run_testset(network, test_batcher, params, 0, equal_mask='nonequal' not in checkpoint)

        result_ssim = []
        for i, (gnd, pred, und, seg) in enumerate(vis_nenm):
            result_ssim.append(ssim(gnd[2], pred[2]))  # Test middle slice (needle best visible)

        arr = np.asarray(result_ssim)

        idx_min = arr.argmin()
        idx_mean = (np.abs(arr - np.mean(arr))).argmin()
        idx_max = arr.argmax()

        logging.info(f'{result_ssim[idx_min]}, {result_ssim[idx_mean]}, {result_ssim[idx_max]}')
        logging.info(repr(arr))

        exit(1)

        name = checkpoint.split('.')[0]
        dirname = path / name
        dirname.mkdir(parents=True, exist_ok=True)

        logging.info('model loaded. Start eval')
        times = []
        for i in range(20):
            with torch.no_grad():
                img = next(test_batcher.items())
                im_und, k_und, mask, im_gnd = prepare_input_as_variable(img,
                                                                        11,
                                                                        params.config.train.undersampling,
                                                                        equal_mask=False)
                t_start = time.time()
                result, _ = network(im_und, k_und, mask, test=False)
                t_end = time.time()
                # plt.imshow(result[0,0,:,:,4].cpu().detach(), cmap="Greys_r", interpolation="nearest", aspect='auto')
                # plt.savefig(dirname / f'res{i}.png')
                if i > 2:
                    times.append(t_end - t_start)
                logging.info(f'actual inference speed: {t_end - t_start}')
        logging.info(f'average actual inference speed (i > 2) {np.average(times)}')

        # run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params, dirname, name, 0, 0, 0, '')
        # exit(1)

def print_comparisons(params: Parameters):
    # path =
    equal_model = params.out_dir / 'equal-exp1_25und_seq5.npz'
    nonequal_model = params.out_dir / 'nonequalnmask-exp1_25und_seq5.npz'

    params.config.data.sequence_length = 5
    params.config.train.undersampling = 25

    dl_test = DataLoader(params.in_dir / 'test')
    dl_test.load(split_regex=params.config.data.split_regex, filter_regex='sag')
    sequencer_test = SequenceBuilder(dl_test)

    kwargs = {'seed': params.config.data.sequence_seed, 'seq_len': params.config.data.sequence_length }
    test_sequences = sequencer_test.generate_singleslice_sequences(**kwargs)

    test_batcher_equal = Batcher(dl_test)
    test_batcher_nonequal = Batcher(dl_test)
    sequence = next(test_sequences.items())
    test_batcher_equal.append_sequence(sequence=sequence,norm=params.config.data.normalize, equal_images=True,
                                    crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x))
    test_batcher_nonequal.append_sequence(sequence=sequence, norm=params.config.data.normalize, equal_images=False,
                                       crop_expand_to=(params.config.data.shape_y, params.config.data.shape_x))
    logging.info('Finished creating test batchers')

    network_equal = CRNNMRI(n_ch=1, nf=64, ks=3, nc=5, nd=5, equal=True).cuda()
    state_dict = torch.load(equal_model)
    network_equal.load_state_dict(state_dict)
    network_equal.eval()

    network_nonequal = CRNNMRI(n_ch=1, nf=64, ks=3, nc=5, nd=5, equal=False).cuda()
    state_dict = torch.load(nonequal_model)
    network_nonequal.load_state_dict(state_dict)
    network_nonequal.eval()

    with torch.no_grad():
        img_eq = next(test_batcher_equal.items())
        im_und_e, k_und_e, mask_e, im_gnd_e = prepare_input_as_variable(img_eq,
                                                                        11,
                                                                        params.config.train.undersampling,
                                                                        equal_mask=True)
        img_neq = next(test_batcher_nonequal.items())
        im_und_ne, k_und_ne, mask_ne, im_gnd_ne = prepare_input_as_variable(img_neq,
                                                                            11,
                                                                            params.config.train.undersampling,
                                                                            equal_mask=False)
        pred_e, _ = network_equal(im_und_e, k_und_e, mask_e, test=False)
        pred_ne, _ = network_nonequal(im_und_ne, k_und_ne, mask_ne, test=False)

        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(f'Exp 1')
        axes = [plt.subplot(2, 3, j + 1) for j in range(6)]

        axes, ax = set_ax(axes, 0, "ground truth", from_tensor_format(im_gnd_e.cpu().numpy())[0][2])
        axes, ax = set_ax(axes, ax, f"{25}x undersampled", from_tensor_format(im_und_e.cpu().numpy())[0][2])
        axes, ax = set_ax(axes, ax, f"reconstructed non-temporal", from_tensor_format(pred_e.cpu().numpy())[0][2])

        axes, ax = set_ax(axes, ax, "ground truth", from_tensor_format(im_gnd_ne.cpu().numpy())[0][2])
        axes, ax = set_ax(axes, ax, f"{25}x undersampled", from_tensor_format(im_und_ne.cpu().numpy())[0][2])
        axes, ax = set_ax(axes, ax, f"reconstructed non-temporal", from_tensor_format(pred_ne.cpu().numpy())[0][2])

        fig.tight_layout()
        plt.savefig(params.out_dir / f'exp1.png', pad_inches=0)
        plt.close(fig)









def sig():
    equal = [0.60093324, 0.60570815, 0.54688763, 0.69313578, 0.57454376,
       0.53080537, 0.51921791, 0.50214974, 0.57018188, 0.60767177,
       0.54975735, 0.52223133, 0.53686644, 0.56803705, 0.54338076,
       0.5878686 , 0.58906052, 0.62367299, 0.61492423, 0.59146942,
       0.63447168, 0.57670919, 0.58243392, 0.57925407, 0.61728971,
       0.65194093, 0.6043309 , 0.51564812, 0.55123598, 0.60080896,
       0.60729956, 0.53548502, 0.59482152, 0.61050679, 0.50596508,
       0.58695029, 0.61772965, 0.55797598, 0.54345939, 0.58955698,
       0.54206024, 0.56711515, 0.5534047 , 0.57690338, 0.60330198,
       0.58364541, 0.60313934, 0.61682053, 0.57214941, 0.62788954,
       0.58130558, 0.56376189, 0.62833117, 0.59277598, 0.56748542,
       0.54060547, 0.63023084, 0.53052595, 0.62874817, 0.54268657,
       0.55625203, 0.53526007, 0.54317359, 0.59387247, 0.61358788,
       0.60599396, 0.47389762, 0.61569693, 0.5410105 , 0.48298613,
       0.62841454, 0.57289907, 0.59517101, 0.51747092, 0.50802844,
       0.49219494, 0.51248198, 0.54277514, 0.60429371, 0.55052976,
       0.59539129, 0.61221745, 0.61861238, 0.58502852, 0.55120105,
       0.57303394, 0.58491657, 0.57880795, 0.55742929, 0.6275569 ,
       0.62600307, 0.606976  , 0.49888027, 0.54863196, 0.64088257,
       0.52435145, 0.60006339, 0.53604182, 0.55614293, 0.58923064,
       0.53931509, 0.60156527, 0.65914804, 0.6014617 , 0.55620357,
       0.70410966, 0.51368621, 0.51002137, 0.54236354, 0.569973  ,
       0.52162664, 0.55067748, 0.60036744, 0.5252708 , 0.44452509,
       0.56983723, 0.62709406, 0.50197336, 0.54497703, 0.66597144,
       0.64071037, 0.67189774, 0.54463631, 0.65837384, 0.6530896 ,
       0.49653353, 0.58511976, 0.57138722, 0.57312629, 0.56867842,
       0.62844663, 0.62388004, 0.55557064, 0.62493924, 0.4887776 ,
       0.54389392, 0.51464895, 0.58439387, 0.55138824, 0.50301688,
       0.60193009, 0.59641305, 0.56281363, 0.52460533, 0.55248789,
       0.52677714, 0.52883405, 0.56821552, 0.56654599, 0.56911077,
       0.5641955 , 0.58167534, 0.60953141, 0.61637507, 0.50656697,
       0.578189  , 0.48246571, 0.56643225, 0.5523306 , 0.53609593,
       0.58857966, 0.64443648, 0.51894789, 0.66051721, 0.61331159,
       0.5302205 , 0.62457202, 0.55117502, 0.5639588 , 0.47863535,
       0.62238044, 0.60262738, 0.51689827, 0.54845351, 0.57733415,
       0.6631938 , 0.56174953, 0.5449646 , 0.55781612, 0.55853984,
       0.61461333, 0.57685382, 0.62177959, 0.66135434, 0.51022435,
       0.63454919, 0.52380424, 0.64513177, 0.61350272, 0.52273757,
       0.70424405, 0.609497  , 0.61212945, 0.6254048 , 0.62024219,
       0.61580579, 0.58750059, 0.65225087, 0.55957321, 0.43938492]

    nonequal = [0.54868113, 0.67772514, 0.65599501, 0.71946131, 0.63769713,
       0.59832114, 0.57439013, 0.62664125, 0.68706793, 0.6631809 ,
       0.66018906, 0.61002345, 0.61532163, 0.5929891 , 0.57444769,
       0.62955479, 0.60867293, 0.6551871 , 0.64096061, 0.63881527,
       0.6683456 , 0.64122497, 0.67437415, 0.60474359, 0.58557176,
       0.69287289, 0.61171695, 0.54607562, 0.63772449, 0.65335949,
       0.60416957, 0.58993256, 0.61264101, 0.64300234, 0.58363484,
       0.60872009, 0.68442361, 0.59934958, 0.60564539, 0.62869054,
       0.59991812, 0.57254972, 0.62924971, 0.64454937, 0.66232162,
       0.62594082, 0.67407782, 0.68604074, 0.66892077, 0.61393887,
       0.65153162, 0.63688498, 0.65423572, 0.63043407, 0.62118795,
       0.56401986, 0.7124135 , 0.5907993 , 0.59686337, 0.57143423,
       0.54634422, 0.62890523, 0.66019052, 0.63446155, 0.63314577,
       0.66309828, 0.57710433, 0.6545163 , 0.62213001, 0.58343427,
       0.62423637, 0.65600833, 0.65119512, 0.64360436, 0.63176561,
       0.6382702 , 0.57252659, 0.51337091, 0.64104684, 0.63054398,
       0.64674348, 0.65357362, 0.610253  , 0.57922095, 0.61003011,
       0.62868368, 0.63436105, 0.59001118, 0.63703167, 0.6695077 ,
       0.686897  , 0.72067554, 0.59981113, 0.60550259, 0.56413568,
       0.64725305, 0.64949149, 0.61085173, 0.55913139, 0.61710849,
       0.64646178, 0.607119  , 0.65935901, 0.63739019, 0.67847204,
       0.68804848, 0.63559501, 0.58145676, 0.6391795 , 0.59838485,
       0.63917333, 0.57630803, 0.66889708, 0.66157136, 0.61986751,
       0.58978033, 0.60143934, 0.61787931, 0.61204318, 0.70791702,
       0.65067408, 0.68752893, 0.63581581, 0.67058482, 0.68179553,
       0.54639757, 0.61724016, 0.58619262, 0.60060121, 0.62714485,
       0.67381987, 0.61284649, 0.6177598 , 0.7292068 , 0.49159512,
       0.64319251, 0.60241472, 0.56849778, 0.60901778, 0.58589853,
       0.63901657, 0.66912035, 0.68129879, 0.61542539, 0.68716994,
       0.60265031, 0.57731829, 0.61301239, 0.61966695, 0.62357084,
       0.63119789, 0.63837903, 0.65294855, 0.69693123, 0.6068705 ,
       0.66313518, 0.55141187, 0.66125829, 0.63098593, 0.59641174,
       0.66262385, 0.62088788, 0.66699406, 0.67722165, 0.61479364,
       0.55153896, 0.67344894, 0.71388582, 0.62901911, 0.61293655,
       0.67525722, 0.61586071, 0.57817007, 0.61491209, 0.60071017,
       0.66424304, 0.64168167, 0.60988546, 0.66896028, 0.64095756,
       0.67211111, 0.61907964, 0.61401632, 0.68335095, 0.60874734,
       0.68933048, 0.63875   , 0.66902695, 0.68048865, 0.62039275,
       0.73590133, 0.60612208, 0.61705958, 0.66976672, 0.63354657,
       0.69642883, 0.71276382, 0.61974558, 0.61254661, 0.56502611]

    logging.info(ttest_ind(nonequal, equal))