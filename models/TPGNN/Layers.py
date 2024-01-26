import torch
import torch.nn as nn
from .SubLayers import TPGNN
from .TransformerLayers import MultiHeadAttention, PositionwiseFeedForward


class ConvExpandAttr(nn.Module):
    '''
    [batch, n_route, n_time, 1] -> [batch, n_route, n_time, n_attr]
    '''

    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        bias
    ):
        super().__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, bias=bias)

    def forward(self, x):
        # [batch, n_route, n_time, 1] -> [batch, 1, n_route, n_time]
        x = x.permute(0, 3, 1, 2)
        # [batch, 1, n_route, n_time] -> [batch, n_attr, n_route, n_time]
        x = self.conv(x)
        # [batch, n_attr, n_route, n_time] -> [batch, n_route, n_time, n_attr]
        x = x.permute(0, 2, 3, 1)
        return x


class SpatioEnc(nn.Module):
    def __init__(
        self,
        n_route,
        n_attr=33,
        normal=True
    ):
        super().__init__()

        self.enc = nn.Parameter(torch.empty(n_route, n_attr))
        nn.init.xavier_uniform_(self.enc.data)
        # self.w = nn.Linear(n_route, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)

    def forward(self, x):
        enc = self.enc
        x = x.permute(0, 2, 1, 3) + enc
        if self.no:
            x = self.norm(x)
        x = x.permute(0, 2, 1, 3)
        return x


class TempoEnc(nn.Module):
    def __init__(
        self,
        n_time,
        n_attr,
        normal=True
    ):
        super().__init__()

        self.time = n_time
        self.enc = nn.Embedding(n_time, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)

    def forward(self, x, start=0, t_left=None):
        length = x.shape[-2]
        if t_left == None:
            enc = self.enc(torch.arange(start, start + length).cuda())
        else:
            enc = self.enc(torch.Tensor(t_left).long().cuda())
        x = x + enc
        if self.no:
            x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=1
    ):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_in, d_in//2),
            nn.ReLU(inplace=True),
            nn.Linear(d_in//2, d_in//4),
            nn.ReLU(inplace=True),
            nn.Linear(d_in//4, d_out)
        )

    def forward(self, x):
        # [batch, n_route, n_his, n_attr] -> [batch, n_route, n_attr, n_his]
        x = x.permute(0, 1, 3, 2)
        # [batch, n_route, n_attr, n_his] -> [batch ,n_route, n_attr, 1]
        output = self.linear(x)
        # [batch ,n_route, n_attr, 1] -> [batch, n_route, 1, n_attr]
        output = output.permute(0, 1, 3, 2)
        return output


class EncoderLayer_stamp(nn.Module):
    def __init__(
        self,
        num_route, history_len, n_attr, n_hid, dis_mat, attn, STstamp, drop_prob, n_c,
        spa_mask, tem_mask
    ):
        super().__init__()

        # cor_mat = opt.cor_mat

        self.tem_attn = MultiHeadAttention(
            attn['head'], n_attr, attn['d_k'], attn['d_v'], attn['drop_prob'])
        self.tem_mask = tem_mask
        assert STstamp['use'], "encoder_stamp requires time stamp as input."
        kt, droprate, temperature = STstamp['kt'], drop_prob, STstamp['temperature']
        self.stgc = TPGNN(n_attr, n_attr, num_route, history_len, dis_mat, kt, n_c, droprate, temperature)
       
        self.pos_ff1 = PositionwiseFeedForward(n_attr, n_hid, drop_prob)
        self.pos_ff2 = PositionwiseFeedForward(n_attr, n_hid, drop_prob)

    def forward(self, x, stamp):
        x = self.tem_attn(x, x, x, self.tem_mask)
        x = self.pos_ff1(x)
        x = self.stgc(x, stamp)
        x = self.pos_ff2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_attr, n_hid, attn, drop_prob,
        slf_mask, mul_mask
    ):
        super().__init__()


        self.slf_attn = MultiHeadAttention(
            attn['head'], n_attr, attn['d_k'], attn['d_v'], attn['drop_prob'])
        self.slf_mask = slf_mask

        self.mul_attn = MultiHeadAttention(
            attn['head'], n_attr, attn['d_k'], attn['d_v'], attn['drop_prob'])
        self.mul_mask = mul_mask

        self.pos_ff = PositionwiseFeedForward(n_attr, n_hid, drop_prob)

        # dis_mat = opt.dis_mat
        # n_his, d_attribute, d_out, d_q, d_c, kt, temperature = opt.TG5['n_his'], opt.TG5['d_attribute'], opt.TG5['d_out'], opt.TG5['d_q'], opt.TG5['d_c'], opt.TG5['kt'], opt.TG5['temperature']
        # self.stgc = temporalGraphConv_ver5(n_his, d_attribute, d_out, dis_mat, d_q, d_c, kt, temperature)

    def forward(self, x, enc_output):
        x = self.mul_attn(x, enc_output, enc_output, self.mul_mask)
        x = self.pos_ff(x)

        # x = self.stgc(x)
        return x
