#!/usr/bin/env python
from __future__ import print_function, division

import os, time
from typing import List, Tuple
from os.path import join

import torch.optim as optim
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

from .utils.kspace import get_rand_exp_decay_mask
from .utils import compressed_sensing as cs
from .utils.metric import complex_psnr

from .cascadenet_pytorch.model_pytorch import *
from .cascadenet_pytorch.dnn_io import to_tensor_format
from .cascadenet_pytorch.dnn_io import from_tensor_format

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def prep_input(im, acc=4.0):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    b, s, y, x = im.shape
    mask = np.zeros(im.shape)
    for b_ in range(b):
        for s_ in range(s):
            mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / acc, 1 / 3)

    im_und, k_und = cs.undersample(im, mask, centred=True, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]


def prepare_case(case: Tuple[str, List[Path]], key: str = "sag", shape: int = 256, T = 15, slice_count = 3, sequence_shift = 2/3) -> np.ndarray:
    ifr = sitk.ImageFileReader()
    study_id, files = case
    files = list(sorted([file for file in files if key in file.name]))
    if len(files) * slice_count < T:
        raise ValueError(f'insufficient data in case {study_id} to reach a sequence of length {T}')
    images = dict()

    volumes, t, rev = [], 0, False
    while t + T <= len(files) * slice_count:
        sequence = []
        for file in files[t // slice_count:(t+T) // slice_count]:
            ifr.SetFileName(str(file))
            images[file] = images.get(file, sitk.GetArrayFromImage(ifr.Execute()).astype('float64'))
            img = images[file]
            # do we guarantee to take the center slice? do we take all 5 slices, or less? just one? what order?
            z, y, x = img.shape
            if z <= slice_count:
                raise ValueError(f'{z} < {slice_count}, cannot split image up')
            split_n = lambda a: ([a // 2 + (1 if a < a % 2 else 0) for _ in range(2)])
            split_x = split_n(np.abs(x - shape))
            split_y = split_n(np.abs(y - shape))

            if x > shape:
                img = img[:, split_x[0]:-split_x[1], :]
            elif x < shape:
                img = np.pad(img, ([0, 0], [0, 0], x), mode='edge')
            if y > shape:
                img = img[:, :, split_y[0]:-split_y[1]]
            elif y < shape:
                img = np.pad(img, ([0, 0], y, [0, 0]), mode='edge')

            # randoms, center, randoms
            slices = np.random.choice(list(set(range(z)).difference([z // 2])), size=slice_count, replace=False)
            slices[len(slices) // 2] = z // 2
            for s in slices:
                sequence.append(img[s, :, :])

        sequence = list(reversed(sequence)) if rev else sequence
        volumes.append(sequence)

        rev = not rev
        t += int(sequence_shift * T)

    # sanity check
    if not all(len(s) == T for s in volumes):
        raise ValueError(f'not all sequences are equal to {T}')
    return np.stack(volumes)


def generate_volumes(data_dir: Path):
    for patient_dir in data_dir.iterdir():
        try:
            if patient_dir.is_dir():
                files = list(patient_dir.iterdir())
                study_ids = {fn.name.split('_')[1] for fn in files}
                for study_id in study_ids:
                    yield study_id, [fn for fn in files if study_id in fn.name]
        except:
            continue


def crnn_mri(args):
    shape = 256
    volumes = generate_volumes(args.data_dir)
    data = np.concatenate([prepare_case(volume, shape=shape, T=args.t, sequence_shift=0.666) for volume in volumes])

    # Project config
    model_name = 'crnn_mri_test' if args.test else 'crnn_mri'
    acc = args.acceleration_factor  # undersampling rate
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    save_fig = args.savefig
    save_every = 5

    # Configure directory info
    save_dir = args.out_dir.absolute()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    data_split = np.array_split(range(len(data)), 10)
    if len(data_split) < 3 and len(data) // 3 >= batch_size:
        raise ValueError('insufficient data')
    train = [data[i] for i in np.concatenate(data_split[2:])]
    test = [data[i] for i in data_split[0]]
    validate = [data[i] for i in data_split[1]]

    # Specify network
    rec_net = CRNN_MRI()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr), betas=(0.5, 0.999))

    # # build CRNN-MRI with pre-trained parameters
    # rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth'))

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()
    else:
        rec_net = rec_net.cpu()
        criterion.cpu()

    i = 0
    for epoch in range(num_epoch):
        t_start = time.time()

        train_err = 0
        train_batches = 0
        for im in iterate_minibatch(train, batch_size, shuffle=True):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            im_u = Variable(im_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(im_gnd.type(Tensor))

            optimizer.zero_grad()
            rec = rec_net(im_u, k_u, mask, test=args.test)
            loss = criterion(rec, gnd)
            loss.backward()
            optimizer.step()

            train_err += loss.item()
            train_batches += 1

            if args.test and train_batches == 20:
                break

        validate_err = 0
        validate_batches = 0
        rec_net.eval()
        for im in iterate_minibatch(validate, batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

            pred = rec_net(im_u, k_u, mask, test=True)
            err = criterion(pred, gnd)

            validate_err += err
            validate_batches += 1

            if args.test and validate_batches == 20:
                break

        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        for im in iterate_minibatch(test, batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)
            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

            pred = rec_net(im_u, k_u, mask, test=True)
            err = criterion(pred, gnd)
            test_err += err
            for im_i, und_i, pred_i in zip(im,
                                           from_tensor_format(im_und.numpy()),
                                           from_tensor_format(pred.data.cpu().numpy())):
                base_psnr += complex_psnr(im_i, und_i, peak='max')
                test_psnr += complex_psnr(im_i, pred_i, peak='max')

            if save_fig and test_batches % save_every == 0:
                vis.append((from_tensor_format(im_gnd.numpy())[0],
                            from_tensor_format(pred.data.cpu().numpy())[0],
                            from_tensor_format(im_und.numpy())[0],
                            from_tensor_format(mask.data.cpu().numpy(), mask=True)[0]))

            test_batches += 1
            if args.test and test_batches == 20:
                break

        t_end = time.time()

        train_err /= train_batches
        validate_err /= validate_batches
        test_err /= test_batches
        base_psnr /= (test_batches * batch_size)
        test_psnr /= (test_batches * batch_size)

        # Then we print the results for this epoch:
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        print(" validation loss:\t{:.6f}".format(validate_err))
        print(" test loss:\t\t{:.6f}".format(test_err))
        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # save the model
        if epoch in [1, 2, num_epoch - 1]:
            if save_fig:
                for im_i, pred_i, und_i, mask_i in vis:
                    im = abs(np.concatenate([und_i[0], pred_i[0], im_i[0], im_i[0] - pred_i[0]], 1))
                    plt.imsave(join(save_dir, 'im{0}_x.png'.format(i)), im, cmap='gray')

                    im = abs(np.concatenate([und_i[..., 0], pred_i[..., 0],
                                             im_i[..., 0], im_i[..., 0] - pred_i[..., 0]], 0))
                    plt.imsave(join(save_dir, 'im{0}_t.png'.format(i)), im, cmap='gray')
                    plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
                               np.fft.fftshift(mask_i[..., 0]), cmap='gray')
                    i += 1

            name = '%s_epoch_%d.npz' % (model_name, epoch)
            torch.save(rec_net.state_dict(), join(save_dir, name))
            print('model parameters saved at %s' % join(os.getcwd(), name))
            print('')
