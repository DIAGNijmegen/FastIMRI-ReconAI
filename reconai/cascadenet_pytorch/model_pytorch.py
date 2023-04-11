import logging
import pathlib

import torch
import numpy as np
import torch.nn as nn

from reconai.cascadenet_pytorch.kspace_pytorch import DataConsistencyInKspace
from reconai.cascadenet_pytorch.module import Module
import matplotlib.pyplot as plt

class CRNNcell(Module):
    """
    Convolutional RNN cell that evolves over both time and iterations
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        """
        Parameters
        ----------
        input: torch.Tensor
            4d tensor, shape (batch_size, channel, width, height)
        hidden: torch.Tensor
            hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
        hidden_iteration: torch.Tensor
            hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

        Returns
        -----------------
        output: torch.Tensor
            4d tensor, shape (batch_size, hidden_size, width, height)
        """
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(Module):
    """
    Bidirectional Convolutional RNN layer
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration):
        """
        Parameters
        ----------
        input: torch.Tensor
            5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
        input_iteration: torch.Tensor
            5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)

        Returns
        -------
        output: torch.Tensor
            5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
        """
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = self.init_hidden(size_h)

        output_f = []
        # output_b = []
        # forward pass
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        # DISABLED BACKWARD PASS

        # output_f = torch.cat(output_f)

        # backward pass
        # hidden = hid_init
        # for i in range(nt):
        #     hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i - 1], hidden)
        #
        #     output_b.append(hidden)
        # output_b = torch.cat(output_b[::-1])
        #
        # output = output_f + output_b

        output = torch.cat(output_f)
        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNNMRI(Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    # incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape


    """

    def __init__(self, n_ch: int = 2, nf: int = 64, ks: int = 3, nc: int = 5, nd: int = 5):
        """
        Parameters
        ----------
            n_ch: int
                number of channels
            nf: int
                number of filters
            ks: int
                kernel size
            nc: int
                number of iterations
            nd: int
                number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNNMRI, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        def conv2d():
            return nn.Conv2d(nf, nf, ks, padding=ks // 2).type(self.TensorType)

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = conv2d()
        self.conv1_h = conv2d()
        self.conv2_x = conv2d()
        self.conv2_h = conv2d()
        self.conv3_x = conv2d()
        self.conv3_h = conv2d()
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding=ks // 2).type(self.TensorType)
        self.relu = nn.LeakyReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, gnd, test=False):
        """
        Parameters
        ----------
        x: torch.Tensor
            input in image domain, of shape (n, 2, nx, ny, n_seq)
        k: torch.Tensor
            initially sampled elements in k-space
        m: torch.Tensor
            corresponding nonzero location
        gnd: torch.Tensor
            groundtruth
        test: bool, optional
            True: the model is in test mode, False: train mode

        Returns
        -------
        output: torch.Tensor
            [output_image] with shape (batch_size, 2, width, height, n_seq)
        """
        net, ti_out, out = {}, '', {}
        logging.debug(f'net init @ {mem_info()}')
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = self.init_hidden(size_h)

        for j in range(self.nd - 1):
            net['t0_x%d' % j] = hid_init

        k = torch.complex(k[:, 0, ...], k[:, 1, ...]).unsqueeze(0)

        for i in range(1, self.nc + 1):
            o = i - 1

            # print_progress_model(gnd, x, 'pre_bcrnn', False)
            x = x.permute(4, 0, 1, 2, 3)
            x = x.contiguous()

            ti_x0, to_x0 = f't{i}_x0', f't{o}_x0'
            net[to_x0] = net[to_x0].view(n_seq, n_batch, self.nf, width, height)
            net[ti_x0] = self.bcrnn(x, net[to_x0])
            net[ti_x0] = net[ti_x0].view(-1, self.nf, width, height)

            # print_progress_model(gnd, net[ti_x0], 'post_bcrnn', True)
            ti_x1, ti_h1, to_x1 = f't{i}_x1', f't{i}_h1', f't{o}_x1'
            net[ti_x1] = self.conv1_x(net[ti_x0])
            net[ti_h1] = self.conv1_h(net[to_x1])
            net[ti_x1] = self.relu(net[ti_h1] + net[ti_x1])

            # print_progress_model(gnd, net[ti_x1], 'post_crnn1', True)
            ti_x2, ti_h2, to_x2 = f't{i}_x2', f't{i}_h2', f't{o}_x2'
            net[ti_x2] = self.conv2_x(net[ti_x1])
            net[ti_h2] = self.conv2_h(net[to_x2])
            net[ti_x2] = self.relu(net[ti_h2] + net[ti_x2])

            # print_progress_model(gnd, net[ti_x2], 'post_crnn2', True)
            ti_x3, ti_h3, to_x3 = f't{i}_x3', f't{i}_h3', f't{o}_x3'
            net[ti_x3] = self.conv3_x(net[ti_x2])
            net[ti_h3] = self.conv3_h(net[to_x3])
            net[ti_x3] = self.relu(net[ti_h3] + net[ti_x3])

            ti_x4 = f't{i}_x4'
            net[ti_x4] = self.conv4_x(net[ti_x3])

            x = x.view(-1, n_ch, width, height)
            ti_out = f't{i}_out'
            net[ti_out] = x + net[ti_x4]

            net[ti_out] = net[ti_out].view(-1, n_batch, n_ch, width, height)
            net[ti_out] = net[ti_out].permute(1, 2, 3, 4, 0)
            net[ti_out].contiguous()
            net[ti_out] = self.dcs[i - 1].perform(net[ti_out], k, m)
            x = net[ti_out]
            # print_progress_model(gnd, x, 'post_dc', False)
            logging.debug(f'it {i} @ {mem_info()}')
            out[ti_out] = x
            # clean up o=i-1
            if test:
                to_delete = [key for key in net if f't{o}' in key]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net[ti_out], out


def print_progress_model(gnd, pred, name, shape: bool):
    fig = plt.figure(figsize=(20, 8))
    axes = [plt.subplot(2, 4, j + 1) for j in range(4 + 2)]

    gnd = gnd.permute(4, 0, 1, 2, 3).cpu()[0][0]
    pred = pred.detach().cpu()
    axes, ax = set_ax(axes, 0, "ground truth", gnd[0])
    # axes, ax = set_ax(axes, ax, f"{1}x undersampled", und[0])
    for k in range(3):
        if shape:
            im = pred.permute(1, 0, 2, 3)[-1]
            axes, ax = set_ax(axes, ax, f"pred {k}", im[k])
        else:
            k_pred = pred.permute(4, 0, 1, 2, 3)
            axes, ax = set_ax(axes, ax, f"pred {k}", k_pred[k, 0, 0])

    fig.tight_layout()
    plt.savefig(pathlib.Path('../data') / f'{name}_seq.png', pad_inches=0)
    plt.close(fig)

def set_ax(axes, ax: int, title: str, image, cmap="Greys_r"):
    axes[ax].set_title(title)
    axes[ax].imshow(np.abs(image), cmap=cmap, interpolation="nearest", aspect='auto')
    axes[ax].set_axis_off()
    return axes, ax + 1

def mem_info():
    gb = 1073741824
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    m = lambda a: np.round(a/gb, decimals=2)
    return f'max: {m(t)} GiB, reserved: {m(r)} GiB, allocated: {m(a)} GiB, free: {m(r - a)} GiB'



