import math
from typing import List

import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

from models.OTPGNET.fsnet_ import DilatedConvEncoder

def laplacian(W):
    N, N = W.shape
    W = W+torch.eye(N).to(W.device)
    D = W.sum(axis=1)
    D = torch.diag(D**(-0.5))
    out = D@W@D
    return out


class TSEncoder(nn.Module):
    def __init__(self, device, num_nodes, supports, history_len, in_dim=2, out_dim=12, residual_channels = 32, dilation_channels = 32, skip_channels=256, end_channels=512, emb_channels = 16, blocks=4, layers=2, dropout=0.3, mask_mode='binomial', gamma=0.9):
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
        kt = 3
        self.reduce_stamp = nn.Linear(history_len, 1, bias=False)  # 平均
        self.temp_1 = nn.Linear(emb_channels, kt + 1)
        self.kt = kt


    def forward(self, x, time_emb, mask):

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        supports = self.supports + [adp]
        #生成矩阵多项式系数
        x = x.transpose(1, 2)
        period_emb = self.reduce_stamp(time_emb.permute(0, 2, 1)).squeeze(2)
        temp_1 = self.temp_1(period_emb)
        '''
        sunpports_new  = []
        for adj in supports:
            adj_new = torch.zeros_like(adj)
            for k in range(self.kt):
                adj_new = adj_new + temp_1[:, k]*torch.matrix_power(adj, k)
            sunpports_new.append(adj_new)
        '''
        x = x.transpose(1, 2)
        x = x.transpose(1, 3)
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.start_conv(x)
        x = self.feature_extractor(x, supports, temp_1)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

