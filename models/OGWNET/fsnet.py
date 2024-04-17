import math
from typing import List

import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

from models.OGWNET.fsnet_ import DilatedConvEncoder



class TSEncoder(nn.Module):
    def __init__(self, device, num_nodes, supports, in_dim=2, out_dim=12, residual_channels = 32, dilation_channels = 32, skip_channels=256, end_channels=512, blocks=4, layers=2, dropout=0.3, mask_mode='binomial', gamma=0.9):
        super().__init__()
        self.mask_mode = mask_mode

        self.supports = supports
        self.supports_len = len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1


        self.feature_extractor = DilatedConvEncoder(
            residual_channels,
            dilation_channels,
            skip_channels,
            blocks,
            layers,
            dropout,
            self.supports_len,
            kernel_size=2,
            gamma=gamma
        )

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, x, mask):
        x = x.transpose(1, 3)
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.start_conv(x)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        supports = self.supports + [adp]
        x = self.feature_extractor(x, supports)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

