import logging
import pathlib

import torch
import numpy as np
import torch.nn as nn

from reconai.model.kspace_pytorch import DataConsistencyInKspace
from reconai.model.module import Module


class CRNNcell(Module):
    """
    Convolutional RNN cell that evolves over both time and iterations
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size,
                             padding=self.kernel_size // 2).type(self.TensorType)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size,
                             padding=self.kernel_size // 2).type(self.TensorType)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size,
                               padding=self.kernel_size // 2).type(self.TensorType)
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
        hid_init = self.init_hidden(size_h)  # 0.0246 s

        output_f = []
        output_b = []
        # forward pass
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)  # 0.0002 s
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward pass
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i - 1], hidden)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNNlayer(Module):
    """
    Convolutional RNN layer
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNlayer, self).__init__()
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

        hidden = hid_init
        input = input.permute(1, 0, 2, 3, 4)[0]  # remove the batch dimension, since CRNN needs 4D input
        input_iteration = input_iteration.permute(1, 0, 2, 3, 4)[0]
        output = self.CRNN_model(input, input_iteration, hidden)  # 0.00023 s

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNNMRI(Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    # incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape


    """

    def __init__(self, n_ch: int, nf: int, ks: int, nc: int, nd: int, bcrnn: bool = True):
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

        assert nd >= 3, "Need at least 3 layers in each iteration"

        def conv2d():
            return nn.Conv2d(nf, nf, ks, padding=ks // 2).type(self.TensorType)

        if bcrnn:
            self.bcrnn = BCRNNlayer(n_ch, nf, ks)
            logging.info('using BCRNN layer')
        else:
            self.bcrnn = CRNNlayer(n_ch, nf, ks)
            logging.info('using single CRNN layer')

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

        # self.hid_in_test = self.init_hidden([5, nf, 256, 256])

    def forward(self, x, k, m, test=False):
        """
        Parameters
        ----------
        x: torch.Tensor
            input in image domain, of shape (1, 1, y, x, s)
        k: torch.Tensor
            initially sampled elements in k-space, of shape (1, 2, y, x, s)
        m: torch.Tensor
            corresponding nonzero location, of shape (1, 1, y, x, s)
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

        hid_init = self.init_hidden(size_h)  # 0.0267 s
        # hid_init = self.hid_in_test

        for j in range(self.nd - 1):
            net[f't0_x{j}'] = hid_init

        k = torch.complex(k[:, 0, ...], k[:, 1, ...]).unsqueeze(0)  # 0.00013 s

        for i in range(1, self.nc + 1):
            o = i - 1

            # t_start_loop = time.time()
            x = x.permute(4, 0, 1, 2, 3)
            x = x.contiguous()

            # t_start = time.time()
            ti_x0, to_x0 = f't{i}_x0', f't{o}_x0'
            net[to_x0] = net[to_x0].view(n_seq, n_batch, self.nf, width, height)
            net[ti_x0] = self.bcrnn(x, net[to_x0])  # 0.055 s bcrnn, 0.016 single crnn
            net[ti_x0] = net[ti_x0].view(-1, self.nf, width, height)

            # t_end = time.time()
            # logging.info(f'bcrnn: {t_end - t_start}')

            # CRNN layers
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
            net[ti_out] = self.dcs[i - 1].perform(net[ti_out], k, m)  # 0.00155 s

            net[ti_out] = torch.clip(net[ti_out], 0, 1)

            x = net[ti_out]
            out[ti_out] = x

            if test:
                for elt in [key for key in net if f't{o}' in key]:
                    del net[elt]

                torch.cuda.empty_cache()

        return net[ti_out], out


def mem_info():
    m = lambda a: np.round(a / 1073741824, decimals=2)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    return f'max: {m(t)} GiB, reserved: {m(r)} GiB, allocated: {m(a)} GiB, free: {m(r - a)} GiB'
