import logging

import torch
import numpy as np
import torch.nn as nn

from reconai.cascadenet_pytorch.kspace_pytorch import DataConsistencyInKspace, AveragingInKspace
from reconai.cascadenet_pytorch.module import Module


def mem_info():
    gb = 1073741824
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    m = lambda a: np.round(a/gb, decimals=2)
    return f'max: {m(t)} GiB, reserved: {m(r)} GiB, allocated: {m(a)} GiB, free: {m(r - a)} GiB'


class DnCn(Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        logging.debug('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = self.conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


class StochasticDnCn(DnCn):
    def __init__(self, n_channels=2, nc=5, nd=5, p=None, **kwargs):
        super(StochasticDnCn, self).__init__(n_channels, nc, nd, **kwargs)

        self.sample = False
        self.p = p
        if not p:
            self.p = np.linspace(0, 0.5, nc)
        logging.debug(self.p)

    def forward(self, x, k, m):
        for i in range(self.nc):
            # stochastically drop connection
            if self.training or self.sample:
                if np.random.random() <= self.p[i]:
                    continue

            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x

    def set_sample(self, flag=True):
        self.sample = flag


class DnCn3D(Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3D, self).__init__()
        self.nc = nc
        self.nd = nd
        logging.debug('Creating D{}C{} (3D)'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = self.conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


class DnCn3DDS(Module):
    def __init__(self, n_channels=2, nc=5, nd=5, fr_d=None, clipped=False, mode='pytorch', **kwargs):
        """

        Parameters
        ----------

        fr_d: frame distance for data sharing layer. e.g. [1, 3, 5]

        """
        super(DnCn3DDS, self).__init__()
        self.nc = nc
        self.nd = nd
        self.mode = mode
        logging.debug('Creating D{}C{}-DS (3D)'.format(nd, nc))
        if self.mode == 'theano':
            logging.debug('Initialised with theano mode (backward-compatibility)')
        conv_blocks = []
        dcs = []
        kavgs = []

        if not fr_d:
            fr_d = list(range(10))
        self.fr_d = fr_d

        conv_layer = self.conv_block

        # update input-output channels for data sharing
        n_channels = 2 * len(fr_d)
        n_out = 2
        kwargs.update({'n_out': 2})

        for i in range(nc):
            kavgs.append(AveragingInKspace(fr_d, i > 0, clipped, norm='ortho'))
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)
        self.kavgs = nn.ModuleList(kavgs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_ds = self.kavgs[i](x, m)
            if self.mode == 'theano':
                # transpose the layes
                x_ds_tmp = torch.zeros_like(x_ds)
                nneigh = len(self.fr_d)
                for j in range(nneigh):
                    x_ds_tmp[:, 2 * j] = x_ds[:, j]
                    x_ds_tmp[:, 2 * j + 1] = x_ds[:, j + nneigh]
                x_ds = x_ds_tmp

            x_cnn = self.conv_blocks[i](x_ds)
            x = x + x_cnn
            x = self.dcs[i](x, k, m)

        return x


class DnCn3DShared(Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3DShared, self).__init__()
        self.nc = nc
        self.nd = nd
        logging.debug('Creating D{}C{}-S (3D)'.format(nd, nc))

        self.conv_block = self.conv_block(n_channels, nd, **kwargs)
        self.dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_block(x)
            x = x + x_cnn
            x = self.dc.perform(x, k, m)

        return x


class CRNNcell(Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2).type(self.TensorType)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = self.init_hidden(size_h)

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i - 1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN_MRI(Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """

    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net, ti_out = {}, ''
        logging.debug(f'net init @ {mem_info()}')
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = self.init_hidden(size_h)

        for j in range(self.nd - 1):
            net['t0_x%d' % j] = hid_init

        k = torch.complex(k[:, 0, ...], k[:, 1, ...]).unsqueeze(0)

        for i in range(1, self.nc + 1):
            o = i - 1

            x = x.permute(4, 0, 1, 2, 3)
            x = x.contiguous()

            ti_x0, to_x0 = f't{i}_x0', f't{o}_x0'
            net[to_x0] = net[to_x0].view(n_seq, n_batch, self.nf, width, height)
            net[ti_x0] = self.bcrnn(x, net[to_x0])
            net[ti_x0] = net[ti_x0].view(-1, self.nf, width, height)

            ti_x1, ti_h1, to_x1 = f't{i}_x1', f't{i}_h1', f't{o}_x1'
            net[ti_x1] = self.conv1_x(net[ti_x0])
            net[ti_h1] = self.conv1_h(net[to_x1])
            net[ti_x1] = self.relu(net[ti_h1] + net[ti_x1])

            ti_x2, ti_h2, to_x2 = f't{i}_x2', f't{i}_h2', f't{o}_x2'
            net[ti_x2] = self.conv2_x(net[ti_x1])
            net[ti_h2] = self.conv2_h(net[to_x2])
            net[ti_x2] = self.relu(net[ti_h2] + net[ti_x2])

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

            logging.debug(f'it {i} @ {mem_info()}')

            # clean up o=i-1
            if test:
                to_delete = [key for key in net if f't{o}' in key]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net[ti_out]
