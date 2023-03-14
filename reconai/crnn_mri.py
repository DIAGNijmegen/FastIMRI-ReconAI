#!/usr/bin/env python
from __future__ import print_function, division

import datetime
import random
import time
import logging
from pathlib import Path
from typing import List
from os.path import join

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from reconai.utils.kspace import get_rand_exp_decay_mask
from reconai.utils import compressed_sensing as cs
from reconai.utils.metric import complex_psnr
from reconai.cascadenet_pytorch.module import Module
from reconai.cascadenet_pytorch.model_pytorch import CRNN_MRI
from reconai.cascadenet_pytorch.dnn_io import to_tensor_format, from_tensor_format
from .utils.kspace import kspace_to_image, image_to_kspace
from .utils.data import prepare_input, iterate_minibatch, generate_datasets, show_images

def iimshow(im):
    plt.imshow(np.abs(im.squeeze()), cmap="Greys_r")
    plt.show()


def prep_input(im, R: int = 4):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    R: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    b, s, y, x = im.shape
    mask = np.zeros(im.shape)
    for b_ in range(b):
        for s_ in range(s):
            mask[b_, s_] = get_rand_exp_decay_mask(y, x, 1 / R, 1 / 3)

    im_und, k_und = cs.undersample(im, mask, centred=True, norm='ortho')
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und, complex=True))
    mask_l = torch.from_numpy(to_tensor_format(mask))
    im_gnd_l = torch.from_numpy(to_tensor_format(im))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def gather_data(data_dir: Path, debug: bool = False):
    data = []
    for patient_dir in data_dir.iterdir():
        try:
            if patient_dir.is_dir():
                files = list(patient_dir.iterdir())
                study_ids = {fn.name.split('_')[1] for fn in files if not fn.name.startswith('tmp')}
                for study_id in study_ids:
                    data.append(Volume(study_id, [fn for fn in files if study_id in fn.name]))
        except:
            continue
        if debug and len(data) > 20:
            break
    return data


class Volume:
    key: str = 'sag'
    shape: int = 256  # intended shape of image (w x h)
    sequence: int = 15  # length of total sequence
    slicing: int = 3  # slices to take from image and add to sequence
    sequence_shift: float = 2/3

    def __init__(self, study_id: str, files: List[Path]):
        self.study_id = study_id
        self.files = files
        self._ifr = sitk.ImageFileReader()
        if not (self.sequence / self.slicing).is_integer():
            raise ValueError(f'{self.sequence}÷{self.slicing} is not an integer')

    def __repr__(self):
        return f'{self.study_id} ({len(self.files)} files)'

    def to_ndarray(self) -> np.ndarray:
        key, sh, T, sc, s_shift = self.key, self.shape, self.sequence, self.slicing, self.sequence_shift
        files = list(sorted([file for file in self.files if key in file.name]))
        if len(files) * sc < T:
            raise ValueError(f'insufficient data in case {self.study_id} to reach a sequence of length {T}')
        images = dict()

        volumes, t, rev = [], 0, False
        while t + T <= len(files) * sc:
            sequence = []
            for file in files[t // sc:(t + T) // sc]:
                self._ifr.SetFileName(str(file))
                images[file] = images.get(file, sitk.GetArrayFromImage(self._ifr.Execute()).astype('float64'))
                img = images[file]
                # do we guarantee to take the center slice? do we take all 5 slices, or less? just one? what order?
                z, y, x = img.shape
                if z < sc:
                    raise ValueError(f'{z} < {sc}, cannot split image up ({file}')
                split_n = lambda a: ([a // 2 + (1 if a < a % 2 else 0) for _ in range(2)])
                split_x = split_n(np.abs(x - sh))
                split_y = split_n(np.abs(y - sh))

                if x > sh:
                    img = img[:, split_x[0]:-split_x[1], :]
                elif x < sh:
                    img = np.pad(img, ([0, 0], [0, 0], x), mode='edge')
                if y > sh:
                    img = img[:, :, split_y[0]:-split_y[1]]
                elif y < sh:
                    img = np.pad(img, ([0, 0], y, [0, 0]), mode='edge')

                # randoms, center, randoms
                slices = np.random.choice(list(set(range(z)).difference([z // 2])), size=sc, replace=False)
                slices[len(slices) // 2] = z // 2
                for s in slices:
                    sequence.append(img[s, :, :])

            sequence = list(reversed(sequence)) if rev else sequence
            volumes.append(sequence)

            rev = not rev
            t += int(s_shift * T)

        # sanity check
        if not all(len(s) == T for s in volumes):
            raise ValueError(f'not all sequences are equal to {T}')
        return np.stack(volumes)


class Batcher:
    batch_size: int = 1
    shuffle: bool = True

    def __init__(self, volumes: List[Volume]):
        self.volumes: List[Volume | np.ndarray] = list(volumes)
        self._blacklist = []

    def get_blacklist(self):
        for _ in self.generate():
            pass
        return self._blacklist

    def generate(self):
        minibatch = []
        all_array = False  # self.volumes only has array elements
        while not (all_array and len(self.volumes) < self.batch_size - len(minibatch)):
            nv = len(self.volumes)

            # convert all self.volumes to arrays when it is small, while loop conditional takes effect
            if not all_array and nv <= self.batch_size:
                for item in self.volumes:
                    if isinstance(item, Volume):
                        self.volumes.remove(item)
                        try:
                            data = item.to_ndarray()
                        except Exception as e:
                            logging.warning(str(e))
                            self._blacklist.append(item.study_id)
                            continue
                        else:
                            self.volumes.extend([data[d] for d in reversed(range(len(data)))])
                all_array = True

            # select a random from volume
            nex = np.random.choice(range(nv)) if Batcher.shuffle else nv - 1
            try:
                item = self.volumes.pop(nex)
            except IndexError as e:
                logging.warning(str(e))
                break

            if isinstance(item, Volume):
                # select random array if volume splits into multiple, rest is put back
                try:
                    data = item.to_ndarray()
                except Exception as e:
                    logging.warning(str(e))
                    self._blacklist.append(item.study_id)
                    continue
                else:
                    nex = np.random.choice(range(len(data))) if Batcher.shuffle else 0
                    for d in reversed(range(len(data))):
                        if d == nex:
                            minibatch.append(data[d])
                        else:
                            self.volumes.append(data[d])
            else:
                data = item
                minibatch.append(data)

            if len(minibatch) == Batcher.batch_size:
                yield np.stack(minibatch)
                minibatch = []


def train(args):
    # change relu to leakyrelu, change loss to 0-1 somehow
    # Project config
    model_name = f'crnn_mri_debug' if args.debug else f'crnn_mri'
    acc = args.acceleration_factor  # undersampling rate
    num_epoch = 3 if args.debug else args.num_epoch
    folds = args.folds if args.folds > 2 else 1
    seg_ai_dir = args.seg_ai_dir

    Batcher.batch_size = args.batch_size
    Volume.key = 'sag'
    Volume.shape = 256
    Volume.sequence = args.sequence_len
    Volume.slicing = 3
    Volume.sequence_shift = 0.666

    data = gather_data(args.in_dir, debug=args.debug)
    data_error = Batcher(data).get_blacklist()
    data = list(filter(lambda a: a.study_id not in data_error, data))
    data_n = len(data)
    logging.info(f"{data_n} volumes found, {len(data_error)} dropped out")
    if data_n < 3:
        raise ValueError('insufficient data for training')

    cuda = True if torch.cuda.is_available() else False
    logging.info("using cuda" if cuda else "using cpu")

    Module.TensorType = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = Module.TensorType

    # Configure directory info
    save_dir: Path = args.out_dir / f'{model_name}_{args.date}'
    save_dir.mkdir(parents=True)
    logging.info(f"saving model to {save_dir.absolute()}")

    # Specify network
    rec_net = CRNN_MRI(n_ch=1, nc=2 if args.debug else 5)
    optimizer = optim.Adam(rec_net.parameters(), lr=float(args.lr), betas=(0.5, 0.999))
    if args.loss == 'ssim':
        segAI = True
        raise NotImplementedError()
    elif args.loss == 'mse+ssim':
        segAI = True
        raise NotImplementedError()
    else:  # 'mse'
        segAI = False
        criterion = torch.nn.MSELoss()

    if segAI and not seg_ai_dir:
        raise ValueError(f'missing {seg_ai_dir} directory')

    if cuda:
        rec_net = rec_net.cuda()
        criterion.cuda()
    else:
        rec_net = rec_net.cpu()
        criterion.cpu()

    data_n = list(range(data_n))
    random.seed(args.seed)
    random.shuffle(data_n)
    data_split = np.array_split(data_n, folds + 1 if folds > 2 else 5)
    logging.info(f'started {folds}-fold training at {datetime.datetime.now()}')
    for fold in range(folds):
        fold_dir = save_dir / f'fold_{fold}'
        k_validation = {fold + 1}
        k_training = set(range(1, len(data_split))).difference(k_validation)
        # k_test is 0
        graph_train_err, graph_val_err = [], []

        for epoch in range(num_epoch):
            t_start = time.time()
            train = Batcher([data[i] for i in np.concatenate([data_split[i] for i in k_training])])
            validate = Batcher([data[i] for i in np.concatenate([data_split[i] for i in k_validation])])
            test = Batcher([data[i] for i in data_split[0]])

            # Training
            train_err = 0
            train_batches = 0
            for im in train.generate():
                logging.debug(f"batch {train_batches}")
                scaler = MinMaxScaler()
                for layer in range(im.shape[1]):
                    im[0, layer, :, :] = scaler.fit_transform(im[0, layer, :, :])
                im_und, k_und, mask, im_gnd = prep_input(im, acc)
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))

                optimizer.zero_grad()
                rec = rec_net(im_u, k_u, mask)
                loss = criterion(rec, gnd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rec_net.parameters(), max_norm=5)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1

                if args.debug and train_batches == 2:
                    break
            logging.info(f"completed {train_batches} train batches")

            validate_err = 0
            validate_batches = 0
            rec_net.eval()
            with torch.no_grad():
                for im in validate.generate():
                    logging.debug(f"batch {validate_batches}")
                    scaler = MinMaxScaler()
                    for layer in range(im.shape[1]):
                        im[0, layer, :, :] = scaler.fit_transform(im[0, layer, :, :])
                    im_und, k_und, mask, im_gnd = prep_input(im, acc)
                    im_u = Variable(im_und.type(Tensor))
                    k_u = Variable(k_und.type(Tensor))
                    mask = Variable(mask.type(Tensor))
                    gnd = Variable(im_gnd.type(Tensor))

                    pred = rec_net(im_u, k_u, mask, test=True)
                    err = criterion(pred, gnd)

                    validate_err += err.item()
                    validate_batches += 1

                    if args.debug and validate_batches == 2:
                        break
            logging.info(f"completed {validate_batches} validate batches")

            vis = []
            base_psnr = 0
            test_psnr = 0
            test_batches = 0
            with torch.no_grad():
                for im in test.generate():
                    logging.debug(f"batch {test_batches}")
                    scaler = MinMaxScaler()
                    for layer in range(im.shape[1]):
                        im[0, layer, :, :] = scaler.fit_transform(im[0, layer, :, :])
                    im_und, k_und, mask, im_gnd = prep_input(im, acc)
                    im_u = Variable(im_und.type(Tensor))
                    k_u = Variable(k_und.type(Tensor))
                    mask = Variable(mask.type(Tensor))

                    pred = rec_net(im_u, k_u, mask, test=True)

                    for im_i, und_i, pred_i in zip(im,
                                                   from_tensor_format(im_und.numpy()),
                                                   from_tensor_format(pred.data.cpu().numpy())):
                        base_psnr += complex_psnr(im_i, und_i, peak='max')
                        test_psnr += complex_psnr(im_i, pred_i, peak='max')

                    if test_batches == 0:  # save n samples
                        vis.append((from_tensor_format(im_gnd.numpy())[0],
                                    from_tensor_format(pred.data.cpu().numpy())[0],
                                    0 if not segAI else 0))

                    test_batches += 1
                    if args.debug and test_batches == 2:
                        break
            logging.info(f"completed {test_batches} test batches")

            t_end = time.time()

            train_err /= train_batches
            validate_err /= validate_batches
            base_psnr /= (test_batches * args.batch_size)
            test_psnr /= (test_batches * args.batch_size)

            stats = '\n'.join([f'Epoch {epoch + 1}/{num_epoch}',
                     f'\ttime: {t_end - t_start} s',
                     f'\ttraining loss:\t\t{train_err}',
                     f'\tvalidation loss:\t\t{validate_err}',
                     f'\tbase PSNR:\t\t\t{base_psnr}',
                     f'\ttest PSNR:\t\t\t{test_psnr}'])
            logging.info(stats)

            graph_train_err.append(train_err)
            graph_val_err.append(validate_err)

            # graph_train_err_norm = [err / np.linalg.norm(graph_train_err, ord=np.inf) for err in graph_train_err]
            # graph_val_err_norm = [err / np.linalg.norm(graph_val_err, ord=np.inf) for err in graph_val_err]

            if epoch > 0:
                graph_x = list(range(len(graph_train_err)))
                fig = plt.figure()
                plt.plot(graph_x, graph_train_err, label="train_loss", lw=1)
                plt.plot(graph_x, graph_val_err, label="val_loss", lw=1)
                plt.legend()
                plt.ylim(bottom=0)
                plt.ylabel(f'{args.loss} loss')
                plt.xlabel("epoch")
                plt.savefig(fold_dir / "progress.png")
                plt.close(fig)

            if epoch % 5 == 0 or epoch > num_epoch - 5:
                name = f'{model_name}_fold_{fold}_epoch_{epoch}'
                npz_name = f'{name}.npz'

                epoch_dir = fold_dir / name
                epoch_dir.mkdir(parents=True, exist_ok=True)
                for i, (gnd, pred, seg) in enumerate(vis):
                    fig = plt.figure()
                    fig.suptitle(f'{name} (val loss: {validate_err})')
                    axes = [plt.subplot(2, 4 if segAI else 3, j + 1) for j in range(4 if segAI else 3 * 2)]
                    ax: int = 0

                    def set_ax(title: str, im, cmap="Greys_r"):
                        nonlocal ax
                        axes[ax].set_title(title)
                        axes[ax].imshow(np.abs(im), cmap=cmap, interpolation="nearest", aspect='auto')
                        axes[ax].set_axis_off()
                        ax = ax + 1

                    norm = lambda a: normalize(a, norm="max")

                    # gnd | pred | err | seg
                    set_ax("ground truth", gnd[0])
                    set_ax("prediction", norm(pred[0]))
                    set_ax("error", norm(gnd[0]) - norm(pred[0]), cmap="magma")

                    gnd_t, pred_t = gnd[..., gnd.shape[-1] // 2], pred[..., pred.shape[-1] // 2]
                    set_ax("ground truth", gnd_t)
                    set_ax("prediction", normalize(pred_t, norm="max"))
                    set_ax("error", gnd_t - pred_t, cmap="magma")

                    fig.tight_layout()
                    plt.savefig(epoch_dir / f'{name}.png', pad_inches=0)
                    plt.close(fig)

                torch.save(rec_net.state_dict(), epoch_dir / npz_name)
                with open(epoch_dir / f'{name}.log', 'w') as log:
                    log.write(stats)

                logging.info(f'fold {fold} model parameters saved at {epoch_dir.absolute()}\n')
    logging.info(f'completed training at {datetime.datetime.now()}')

def normalization_test(args):
    def show_im(original, minmaxscaled, standardscaled, sampledimageminmax, sampledimagestandard):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(original[0])
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(minmaxscaled[0])
        axs[0, 1].set_title('MinMaxScaled')
        axs[0, 2].imshow(sampledimageminmax)
        axs[0, 2].set_title('Sampled')
        axs[0, 3].imshow(sampledimageminmax - minmaxscaled[0])
        axs[0, 3].set_title('MM - S diff')
        axs[1, 0].imshow(original[0])
        axs[1, 0].set_title('Original Image')
        axs[1, 1].imshow(standardscaled[0])
        axs[1, 1].set_title('StandardScaled')
        axs[1, 2].imshow(sampledimagestandard)
        axs[1, 2].set_title('Sampled')
        axs[1, 3].imshow(sampledimagestandard - standardscaled[0])
        axs[1, 3].set_title('SS - S diff')
        plt.show()
    train, _, _ = generate_datasets(args.batch_size, args.sequence_len, args.in_dir)

    image = next(iterate_minibatch(train, args.batch_size, shuffle=True))

    # MINMAX Scaling
    scaler = MinMaxScaler()
    minmax_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        minmax_scaled_image[0, layer, :, :] = scaler.fit_transform(minmax_scaled_image[0, layer, :, :])
    # minmax_scaled_image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    # Standard Scaling

    scaler = StandardScaler()
    standard_scaled_image = image.copy()
    for layer in range(image.shape[1]):
        standard_scaled_image[0, layer, :, :] = scaler.fit_transform(standard_scaled_image[0, layer, :, :])
    # standard_scaled_image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    #
    # im_und, k_und, mask, im_gnd = prepare_input(minmax_scaled_image, 1)
    # und_array_mm = np.asarray(from_tensor_format(im_und.detach().cpu(), True))
    #
    # im_und, k_und, mask, im_gnd = prepare_input(standard_scaled_image, 1)
    # und_array_ss = np.asarray(from_tensor_format(im_und.detach().cpu(), True))
    sampled_mm = kspace_to_image(image_to_kspace(minmax_scaled_image[0][0]))
    sampled_ss = kspace_to_image(image_to_kspace(standard_scaled_image[0][0]))
    show_im(image[0], minmax_scaled_image[0], standard_scaled_image[0], sampled_mm, sampled_ss)

    exit(1)
