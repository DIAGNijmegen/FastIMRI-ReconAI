from typing import List

import torch.nn as nn
import torch
from torch.autograd import Variable


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


class Module(nn.Module):
    TensorType = torch.cuda.FloatTensor

    def __new__(cls, *args, **kwargs):
        if not cls.TensorType:
            raise ValueError('TensorType is not set!')
        return super(Module, cls).__new__(cls)

    def __init__(self):
        super(Module, self).__init__()
        self.cuda_available = True if torch.cuda.is_available() else False

    def init_hidden(self, size: List[int]):
        if self.cuda_available:
            return Variable(torch.zeros(size)).type(self.TensorType).cuda()
        else:
            return Variable(torch.zeros(size)).type(self.TensorType).cpu()

    @staticmethod
    def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):
        # convolution dimension (2D or 3D)
        if conv_dim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # output dim: If None, it is assumed to be the same as n_ch
        if not n_out:
            n_out = n_ch

        # dilated convolution
        pad_conv = 1
        if dilation > 1:
            # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
            # pad = dilation
            pad_dilconv = dilation
        else:
            pad_dilconv = pad_conv

        def conv_i():
            return conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True).type(Module.TensorType)

        conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True).type(Module.TensorType)
        conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True).type(Module.TensorType)

        # relu
        nll = relu if nl == 'relu' else lrelu

        layers = [conv_1, nll()]
        for i in range(nd - 2):
            if bn:
                layers.append(nn.BatchNorm2d(nf))
            layers += [conv_i(), nll()]

        layers += [conv_n]

        return nn.Sequential(*layers)
