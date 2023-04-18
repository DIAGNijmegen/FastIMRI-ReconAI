import torch.nn as nn
import torch
from reconai.models.kiki.layer_utils import fftshift2, GenConvBlock, DataConsist, ifft2, fft2

class KIKI(nn.Module):
    def __init__(self, iters, k, in_ch, out_ch, fm, i):
        super(KIKI, self).__init__()

        conv_blocks_K = [] 
        conv_blocks_I = []
        
        for j in range(iters):
            conv_blocks_K.append(GenConvBlock(k, in_ch, out_ch, fm))
            conv_blocks_I.append(GenConvBlock(i, in_ch, out_ch, fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = iters

    def forward(self, kspace_us, mask):
        # kspace_us = torch.complex(kspace_us[:, 0, ...], kspace_us[:, 1, ...])
        # rec = kspace_us
        rec = fftshift2(kspace_us)
        for i in range(self.n_iter):
            rec = self.conv_blocks_K[i](rec)
#            rec = DataConsist(fftshift2(rec), kspace_us, mask, is_k = True)
            rec = fftshift2(rec)
            rec = ifft2(rec)
            rec = rec + self.conv_blocks_I[i](rec)
            rec = DataConsist(rec, kspace_us, mask)
            
            if i < self.n_iter - 1:
                rec = fftshift2(fft2(rec))
        
        return rec
