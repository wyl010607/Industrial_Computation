import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Layers import ConvExpandAttr, SpatioEnc, TempoEnc, MLP, EncoderLayer_stamp, DecoderLayer

import random


class SrcProcess(nn.Module):
    def __init__(self, history_len, num_route, n_attr, CE, LE, SE, TE, dis_mat):
        super().__init__()

        # n_attr = 33

        self.CE = CE['use']
        if self.CE:
            self.enc_exp = ConvExpandAttr(
                1, n_attr, CE['kernel_size'], CE['bias'])

        self.LE = LE['use']
        if self.LE:
            self.enc_exp = nn.Linear(1, n_attr, bias=LE['bias'])

        self.SE = SE['use']

        if self.SE:
            self.enc_spa_enco = SpatioEnc(num_route, n_attr, SE['nor'])

        self.TE = TE['use']
        if self.TE:
            self.enc_tem_enco = TempoEnc(history_len, n_attr, TE['nor'])
        # self.time_emb = nn.Embedding(opt.circle, opt.n_attr//4)
        # self.emb_conv = nn.Linear(opt.n_attr//4, opt.n_attr, bias=False)
        self.distant_mat = dis_mat
        # self.re = nn.Linear(64, n_attr)

    def forward(self, src):
        src = self.enc_exp(src)
        b, n, t, k = src.shape
        if self.SE:
            src = self.enc_spa_enco(src)
        if self.TE:
            # src = src+self.emb_conv(self.time_emb(stamp)).reshape(b, 1, t, k)
            src = self.enc_tem_enco(src)

        return src


class TrgProcess(nn.Module):
    def __init__(self, history_len, forcest_len, num_route, n_attr, CE, SE, TE, T4N
                 ):
        super().__init__()

        self.mlp = MLP(history_len, 1)

        self.CE = CE['use']
        if self.CE:
            self.dec_exp = ConvExpandAttr(
                1, n_attr, CE['kernel_size'], CE['bias'])

        # spatio encoding
        self.SE = SE['use']
        if self.SE:
            self.dec_spa_enco = SpatioEnc(num_route, n_attr, SE['nor'])

        # temporal encoding
        self.TE = TE['use']
        if self.TE:
            self.dec_tem_enco = TempoEnc(
                forcest_len + T4N['step'], n_attr, TE['nor'])

    def forward(self, trg, enc_output, head=None):
        head = self.mlp(enc_output)
        trg = self.dec_exp(trg)
        if self.SE:
            trg = self.dec_spa_enco(trg)
        trg = torch.cat([head, trg], dim=2)
        if self.TE:
            trg = self.dec_tem_enco(trg)

        return trg


class Decoder(nn.Module):
    def __init__(
            self,
            n_attr, n_hid, attn, drop_prob, n_layer,
            dec_slf_mask, dec_mul_mask
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_attr, n_hid, attn, drop_prob, dec_slf_mask, dec_mul_mask)
            for _ in range(n_layer)
        ])

    def forward(self, x, enc_output):
        for layer in self.layer_stack:
            x = layer(x, enc_output)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            num_route, history_len, n_attr, n_hid, dis_mat, attn, STstamp, drop_prob, n_c,
            enc_spa_mask, enc_tem_mask
    ):
        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer_stamp(num_route, history_len, n_attr, n_hid, dis_mat, attn, STstamp, drop_prob, n_c,
                               enc_spa_mask, enc_tem_mask)
            for _ in range(1)
        ])

    def forward(self, x, time_stamp):
        for layer in self.layer_stack:
            x = layer(x, time_stamp)
        return x


class timestamp(nn.Module):
    def __init__(
            self, circle, n_attr, history_len, TE
    ):
        super().__init__()
        self.time_stamp = nn.Embedding(circle, n_attr // 4)
        # add temporal embedding and normalize
        self.tempral_enc = TempoEnc(history_len, n_attr // 4, TE['nor'])

    def forward(self, stamp):
        time_emb = self.time_stamp(stamp)
        time_emb = self.tempral_enc(time_emb)
        return time_emb


class CSTAGNN_G_stamp(nn.Module):
    def __init__(
            self, n_layer, n_attr, n_hid, reg_A, circle, drop_prob, n_c, a, n_mask, adj_mx, CE, LE, SE, TE, attn,
            STstamp, T4N,
            num_route, history_len, forecast_len
    ):
        super().__init__()

        enc_spa_mask = torch.ones(1, 1, num_route, num_route).cuda()
        enc_tem_mask = torch.ones(1, 1, history_len, history_len).cuda()
        dec_slf_mask = torch.tril(torch.ones(
            (1, 1, forecast_len + 1, forecast_len + 1)), diagonal=0).cuda()
        dec_mul_mask = torch.ones(1, 1, forecast_len + 1, history_len).cuda()

        self.src_pro = SrcProcess(history_len, num_route, n_attr, CE, LE, SE, TE, adj_mx)
        self.trg_pro = TrgProcess(history_len, forecast_len, num_route, n_attr, CE, SE, TE, T4N)
        self.stamp_emb = timestamp(circle, n_attr, history_len, TE)
        self.dec_rdu = ConvExpandAttr(
            n_attr, 1, CE['kernel_size'], CE['bias'])

        self.encoder = Encoder(num_route, history_len, n_attr, n_hid, adj_mx, attn, STstamp, drop_prob, n_c,
                               enc_spa_mask, enc_tem_mask)
        self.decoder = Decoder(n_attr, n_hid, attn, drop_prob, n_layer, dec_slf_mask, dec_mul_mask)

        self.reg_A = reg_A
        self.T4N = T4N['use']
        if self.T4N:
            self.T4N_step = T4N['step']
            self.change_head = T4N['change_head']
            self.change_enc = T4N['change_enc']
            self.T4N_end = T4N['end_epoch']

        self.forecast_len = forecast_len
        self.n_route = num_route
        self.a = a
        self.n_mask = n_mask
        self.n_c = n_c
        #self.apply(self.weights_init)

    '''def weights_init(self, m):
        try:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        except:
            pass'''

    def forward(self, src, time_stamp, trg):
        src = src.permute(0, 2, 1, 3)
        trg = trg.permute(0, 2, 1, 3)
        enc_input = self.src_pro(src)
        time_emb = self.stamp_emb(time_stamp)
        enc_output = self.encoder(enc_input, src)

        dec_input = self.trg_pro(trg, enc_output)
        dec_output = self.decoder(dec_input, enc_output)
        dec_output = self.dec_rdu(dec_output)
        return dec_output.permute(0, 2, 1, 3)
