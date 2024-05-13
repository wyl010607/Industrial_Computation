from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .cross_encoder import Encoder
from .cross_decoder import Decoder
from .cross_embed import DSW_embedding
from .BIAS import BIAS
from .GCN import GCN


class Crossformer(nn.Module):
    def __init__(
        self,
        history_len,
        forecast_len,
        num_nodes,
        seg_len,
        adj_mx,
        forecast_length=1,
        channel=1,
        win_size=4,
        factor=10,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        e_layers=3,
        dropout=0.0,
        baseline=False,
        bias_block=False,
        **kwargs,
    ):
        super(Crossformer, self).__init__()
        self.data_dim = num_nodes
        self.history_len = history_len
        self.forecast_len = forecast_length
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline
        self.adj_mx = nn.Parameter(GCN.build_adj_matrix(adj_mx, adj_type="cheb", K=2))
        self.bias_block = bias_block
        self.bias_forecast_len = forecast_len
        self.para = 5e-2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * history_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * forecast_length / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.history_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, num_nodes, (self.pad_in_len // seg_len), d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=(self.pad_in_len // seg_len),
            factor=factor,
        )

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, num_nodes, (self.pad_out_len // seg_len), d_model)
        )
        self.decoder = Decoder(
            seg_len,
            e_layers + 1,
            d_model,
            n_heads,
            d_ff,
            dropout,
            out_seg_num=(self.pad_out_len // seg_len),
            factor=factor,
        )
        self.Linear = nn.Linear(1, self.history_len)
        self.using_improve = False

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        x_seq = history_data[:, :, :, 0]  # (batch_size, history_len, num_nodes)
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        if x_seq.dim() == 3:
            x = x_seq.unsqueeze(3)
        
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat(
                (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1
            )

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch_size,
        )
        predict_y = self.decoder(dec_in, enc_out)
        pred = (
            base + predict_y[:, : self.forecast_len, :]
        )  # (batch_size, forecast_len, num_nodes)

        input_bias = x 
        if self.using_improve:
            y_res = self.Linear(pred.unsqueeze(-1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            y_res = (y_res + x) * self.para
            input_bias = y_res
        b_s, h_l, n_d, c = input_bias.size()
        if self.bias_block:
            bb = BIAS(b_s, h_l, n_d, c, self.bias_forecast_len, 2, 3, self.adj_mx)
            bias = bb(input_bias)
            # bias = torch.randn(32, 20, 37, 1)
        else:
            bias = torch.zeros(b_s, self.bias_forecast_len, n_d, c).cuda()

        return pred.unsqueeze(-1), bias
