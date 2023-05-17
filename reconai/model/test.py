import numpy as np
import logging
import reconai.utils.metric
import torch

from reconai.model.module import Module
from torch.autograd import Variable
from reconai.data.data import prepare_input
from reconai.model.dnn_io import from_tensor_format
from reconai.utils.metric import complex_psnr
from reconai.utils.graph import print_end_of_epoch


# TODO: Opschonen en herschrijven, evt proberen wat minder duplicate code
def run_and_print_full_test(network, test_batcher_equal, test_batcher_non_equal, params,
                            epoch_dir, name, train_err, validate_err, stats: str = ''):
    sequence_length = params.config.data.sequence_length
    undersampling = params.config.train.undersampling
    iterations = params.config.model.iterations

    vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e = \
        run_testset(network, test_batcher_equal, params, equal_mask=True)
    vis_enm, iters_enm, base_psnr_enm, test_psnr_enm, test_batches_enm = \
        run_testset(network, test_batcher_equal, params, equal_mask=False)
    vis_ne, iters_ne, base_psnr_ne, test_psnr_ne, test_batches_ne = \
        run_testset(network, test_batcher_non_equal, params, equal_mask=True)
    vis_nenm, iters_nenm, base_psnr_nenm, test_psnr_nenm, test_batches_nenm = \
        run_testset(network, test_batcher_non_equal, params, equal_mask=False)
    logging.info(f"completed {test_batches_e} test batches")

    base_psnr_e /= (test_batches_e * params.batch_size)
    base_psnr_enm /= (test_batches_enm * params.batch_size)
    base_psnr_ne /= (test_batches_ne * params.batch_size)
    base_psnr_nenm /= (test_batches_nenm * params.batch_size)
    test_psnr_e /= (test_batches_e * params.batch_size)
    test_psnr_enm /= (test_batches_enm * params.batch_size)
    test_psnr_ne /= (test_batches_ne * params.batch_size)
    test_psnr_nenm /= (test_batches_nenm * params.batch_size)

    stats_psnr = '\n'.join([
        f'\tbase PSNR equal images:\t\t\t{base_psnr_e}',
        f'\tbase PSNR equal images n-masks:\t\t\t{base_psnr_enm}',
        f'\tbase PSNR non-equal images:\t\t\t{base_psnr_ne}',
        f'\tbase PSNR non-equal images n-masks:\t\t\t{base_psnr_nenm}',
        f'\ttest PSNR equal images:\t\t\t{test_psnr_e}',
        f'\ttest PSNR equal images n-masks:\t\t\t{test_psnr_enm}',
        f'\ttest PSNR non-equal images:\t\t\t{test_psnr_ne}',
        f'\ttest PSNR non-equal images n-masks:\t\t\t{test_psnr_nenm}'
    ])
    logging.info(stats_psnr)

    # Loop through vis and gather mean, min and max SSIM to print each
    ssim_e, min_e, mean_e, max_e = print_eoe(vis_e, epoch_dir, f'{name}_equal', validate_err,
                                             sequence_length, undersampling, iters_e, iterations)
    ssim_enm, min_enm, mean_enm, max_enm = print_eoe(vis_enm, epoch_dir, f'{name}_equal_nmasks', validate_err,
                                             sequence_length, undersampling, iters_enm, iterations)
    ssim_ne, min_ne, mean_ne, max_ne = print_eoe(vis_ne, epoch_dir, f'{name}_nonequal', validate_err,
                                                 sequence_length, undersampling, iters_ne, iterations)
    ssim_nenm, min_nenm, mean_nenm, max_nenm = print_eoe(vis_nenm, epoch_dir, f'{name}_nonequal_nmasks',
                                                         validate_err, sequence_length, undersampling,
                                                         iters_nenm, iterations)

    def ft(in_float: float) -> str:
        return format(in_float, '.4f').replace('.', ',')

    with open(epoch_dir / f'{name}.log', 'w') as log:
        log.write(stats)
        log.write(stats_psnr)
        log.write('\n\n SSIM - Equal Images \n')
        log.write(f'MIN: {min_e} | MEAN: {mean_e} | MAX: {max_e} \n\n')
        log.write(np.array2string(ssim_e, separator=','))
        log.write('\n\n SSIM - Equal Images N-Masks \n')
        log.write(f'MIN: {min_enm} | MEAN: {mean_enm} | MAX: {max_enm} \n\n')
        log.write(np.array2string(ssim_enm, separator=','))
        log.write('\n\n SSIM - Non-Equal Images \n')
        log.write(f'MIN: {min_ne} | MEAN: {mean_ne} | MAX: {max_ne} \n\n')
        log.write(np.array2string(ssim_ne, separator=','))
        log.write('\n\n SSIM - Non-Equal Images N-Masks\n')
        log.write(f'MIN: {min_nenm} | MEAN: {mean_nenm} | MAX: {max_nenm} \n\n')
        log.write(np.array2string(ssim_nenm, separator=','))
        log.write('\n\n train_err validate_err min_e mean_e max_e ... enm ... ne ... nenm')
        log.write(f"\n\n\n{ft(train_err)}\t {ft(validate_err)}"
                  f"\t {ft(min_e)}\t {ft(mean_e)}\t{ft(max_e)}"
                  f"\t {ft(min_enm)}\t {ft(mean_enm)}\t{ft(max_enm)}"
                  f"\t {ft(min_ne)}\t {ft(mean_ne)}\t{ft(max_ne)}"
                  f"\t {ft(min_nenm)}\t {ft(mean_nenm)}\t{ft(max_nenm)}")


def run_testset(network, batcher, params, equal_mask: bool):
    vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e = [], [], 0, 0, 0

    with torch.no_grad():
        for im in batcher.items():
            logging.debug(f"batch {test_batches_e}")
            im_und, k_und, mask, im_gnd = prepare_input(im,
                                                        params.config.train.mask_seed,
                                                        params.config.train.undersampling,
                                                        equal_mask=equal_mask)
            im_u = Variable(im_und.type(Module.TensorType))
            k_u = Variable(k_und.type(Module.TensorType))
            mask = Variable(mask.type(Module.TensorType))

            pred, full_iterations = network(im_u, k_u, mask, im_gnd, test=True)

            for im_i, und_i, pred_i in zip(im,
                                           from_tensor_format(im_und.numpy()),
                                           from_tensor_format(pred.data.cpu().numpy())):
                base_psnr_e += complex_psnr(im_i, und_i, peak='max')
                test_psnr_e += complex_psnr(im_i, pred_i, peak='max')

            vis_e.append((from_tensor_format(im_gnd.numpy())[0],
                          from_tensor_format(pred.data.cpu().numpy())[0],
                          from_tensor_format(im_und.numpy())[0],
                          0))
            iters_e.append((from_tensor_format(im_gnd.numpy())[-1],
                            full_iterations))

            test_batches_e += 1
            if params.debug and test_batches_e == 2:
                break
    return vis_e, iters_e, base_psnr_e, test_psnr_e, test_batches_e


def print_eoe(vis, epoch_dir, name, validate_err, sequence_length, undersampling, iters, iterations):
    result_ssim = []
    for i, (gnd, pred, und, seg) in enumerate(vis):
        result_ssim.append(reconai.utils.metric.ssim(gnd[-1], pred[-1]))

    arr = np.asarray(result_ssim)

    idx_min = arr.argmin()
    idx_mean = (np.abs(arr - np.mean(arr))).argmin()
    idx_max = arr.argmax()

    print_end_of_epoch(epoch_dir, [vis[idx_min]], f'{name}_worst', validate_err, result_ssim[idx_min],
                       sequence_length, undersampling, iters[idx_min][0], iters[idx_min][1], iterations)
    print_end_of_epoch(epoch_dir, [vis[idx_mean]], f'{name}_average', validate_err,
                       result_ssim[idx_mean], sequence_length, undersampling, iters[idx_mean][0],
                       iters[idx_mean][1], iterations)
    print_end_of_epoch(epoch_dir, [vis[idx_max]], f'{name}_best', validate_err,
                       result_ssim[idx_max], sequence_length, undersampling, iters[idx_max][0],
                       iters[idx_max][1], iterations)
    return arr, result_ssim[idx_min], result_ssim[idx_mean], result_ssim[idx_max]