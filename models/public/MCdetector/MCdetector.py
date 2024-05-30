import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from .RevIN import RevIN
from tkinter import _flatten
from einops.layers.torch import Rearrange
from einops import reduce
from .embed import PositionalEmbedding


def softmax(f):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))


def get_activation(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "gelu":
        return nn.GELU()
    elif activ == "leaky_relu":
        return nn.LeakyReLU()
    elif activ == "none":
        return nn.Identity()
    else:
        raise ValueError(f"activation:{activ}")


def get_norm(norm, c):
    if norm == 'bn':
        norm_class = nn.BatchNorm2d(c)
    elif norm == 'in':
        norm_class = nn.InstanceNorm2d(c)
    elif norm == 'ln':
        norm_class = nn.LayerNorm(c)
    else:
        norm_class = nn.Identity()

    return norm_class


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        in_features,
        hid_features,
        out_features,
        activ="gelu",
        drop = 0.00,
        jump_conn="proj",
        norm='ln'
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.norm =norm
        self.net = nn.Sequential(
            get_norm(self.norm, in_features),
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            get_norm(self.norm, hid_features),
            nn.Linear(hid_features, out_features),
            nn.Dropout(drop),
        )
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == "proj":
            self.jump_net = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x) + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


class PatchMLP_layer(nn.Module):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn, patch_size, hid_pch, d_model, norm=None, activ="gelu", drop=0.0, jump_conn='proj'):
        super().__init__()
        self.patch_num_mix = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ, drop, jump_conn=jump_conn)
        self.patch_size_mix = MLPBlock(2, patch_size, hid_pch, patch_size, activ, drop,jump_conn=jump_conn)
        self.re_mixing = MLPBlock(3, d_model, d_model, d_model, activ, drop, jump_conn=jump_conn)

        self.norm1 = get_norm(norm, in_chn)
        self.norm2 = get_norm(norm, out_chn)

    def forward(self, x_patch_num, x_patch_size):
        # B C N P
        x_patch_num = self.norm1(x_patch_num)
        x_patch_num = self.patch_num_mix(x_patch_num)
        x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.re_mixing(x_patch_num)

        x_patch_size = self.norm1(x_patch_size)
        x_patch_size = self.patch_size_mix(x_patch_size)
        x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.re_mixing(x_patch_size)

        return x_patch_num, x_patch_size


class MCEncoder(nn.Module):
    def __init__(self, enc_layers):
        super(MCEncoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.num_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1), nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.size_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1) , nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.softmax = nn.Softmax(-1)

    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []
        for enc in self.enc_layers:
            x_patch_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)
            num_logi_list.append(x_patch_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))
            x_patch_num_dist = self.softmax(x_patch_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)
            x_patch_num_dist = reduce(
                x_patch_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_pach_num_dist = rearrange(x_patch_num_dist, "b n p -> b (n p) 1")
            x_patch_size_dist = rearrange(x_patch_size_dist, "b p n -> b (p n) 1")
            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)
        return num_dist_list, size_dist_list, num_logi_list, size_logi_list
    

class Ensemble_block(nn.Module):
    def __init__(self, e_layers):
        super().__init__()
        self.mix_layer = nn.parameter.Parameter(torch.ones(e_layers), requires_grad=True)
    
    def forward(self, dist_list):
        # list of B N D
        dist_list = torch.stack(dist_list, dim=-1)
        # 动态分配权重
        weights = torch.softmax(self.mix_layer, dim=0)
        # 将权重应用至列表中
        dist_list = dist_list * weights
        dist_list = torch.split(dist_list, 1, dim=3)
        dist_list = [t.squeeze(3) for t in dist_list]
        return dist_list


class Encoder_Ensemble(nn.Module):
    def __init__(self, enc_layers ):
        super(Encoder_Ensemble, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.num_mix_layer = Ensemble_block(len(enc_layers))
        self.size_mix_layer = Ensemble_block(len(enc_layers))
        self.softmax = nn.Softmax(-1)

    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []
        T_num_logi_list =[]
        T_size_logi_list = []
        for enc in self.enc_layers:
            x_patch_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)
            x_patch_num = torch.relu(x_patch_num)
            x_patch_size = torch.relu(x_patch_size)
            T_num_logi_list.append(x_patch_num_dist)
            T_size_logi_list.append(x_patch_size_dist)
            num_logi_list.append(x_patch_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))
            x_pach_num_dist = self.softmax(x_patch_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)
            x_pach_num_dist = reduce(
                x_pach_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)
        num_dist_list = self.num_mix_layer(num_dist_list)
        size_dist_list = self.size_mix_layer(size_dist_list)
        return num_dist_list, size_dist_list, num_logi_list, size_logi_list, T_num_logi_list, T_size_logi_list


class MCdetector(nn.Module):
    def __init__(
        self,
        window,
        d_model=50,
        e_layer=3,
        patch_sizes=[3,5,7],
        dropout=0.0,
        activation="gelu",
        channel=55,
        norm='ln',
        output_attention=True,
    ):
        super(MCdetector, self).__init__()
        self.patch_sizes = patch_sizes
        self.window = window
        self.output_attention = output_attention
        self.win_emb = PositionalEmbedding(channel)
        self.patch_num_emb = nn.ModuleList(
            [nn.Linear(patch_size,d_model) for patch_size in patch_sizes]
        )
        self.patch_size_emb = nn.ModuleList(
            [nn.Linear(window//patch_size, d_model) for patch_size in patch_sizes]
        )
        self.patch_encoders = nn.ModuleList()
        self.patch_num_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ=activation, drop=dropout, jump_conn='trunc'),nn.Softmax(-1))
        self.patch_size_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ=activation, drop=dropout, jump_conn='trunc'),nn.Softmax(-1))
        for i, p in enumerate(patch_sizes):
            # 多尺度patch核处理
            patch_size = patch_sizes[i]
            patch_num = self.window // patch_size
            enc_layers = [
                PatchMLP_layer(self.window, 40, channel, int(channel*1.2), int(channel*1.), patch_size, int(patch_size*1.2), d_model, norm, activation, dropout, jump_conn='proj')
                for i in range(e_layer)
            ]
            enc = Encoder_Ensemble(enc_layers=enc_layers)
            self.patch_encoders.append(enc)
        self.recons_num = []
        self.recons_size = []
        for i, p in enumerate(patch_sizes):
            patch_size = patch_sizes[i]
            patch_num = window // patch_size
            self.recons_num.append(nn.Sequential(Rearrange('b c n p -> b c (n p)'), nn.LayerNorm(patch_num*d_model),  nn.Linear(patch_num*d_model, d_model), nn.Sigmoid(), nn.LayerNorm(d_model), nn.Linear(d_model, window), Rearrange('b c l -> b l c')))
            self.recons_size.append(nn.Sequential(Rearrange('b c n p -> b c (n p)'), nn.LayerNorm(patch_size*d_model),  nn.Linear(patch_size*d_model, d_model), nn.Sigmoid(), nn.LayerNorm(d_model), nn.Linear(d_model, window), Rearrange('b c l -> b l c')))
        self.recons_num = nn.ModuleList(self.recons_num)
        self.recons_size = nn.ModuleList(self.recons_size)
        self.rec_alpha = nn.Parameter(torch.zeros(patch_size), requires_grad=True)
        self.rec_alpha.data.fill_(0.5)

    def forward(self, x, mask=None, del_inter=0, del_intra=0):
        B, L, M = x.shape
        patch_num_distribution_list = []
        patch_size_distribution_list = []
        patch_num_mx_list = []
        patch_size_mx_list = []
        revin_layer = RevIN(num_features=M).to(x.device)
        # 实例归一化
        x = revin_layer(x, "norm")
        rec_x = None
        for patch_index, patchsize in enumerate(self.patch_sizes):
            patch_enc = self.patch_encoders[patch_index]
            x = x + self.win_emb(x)
            # x = self.win_emb(x)
            x_patch_num = x_patch_size = x
            # B L C
            x_patch_num = rearrange(x_patch_num, "b (n p) c -> b c n p", p=patchsize)
            x_patch_size = rearrange(x_patch_size, "b (p n) c-> b c p n", p=patchsize)
            x_patch_num = self.patch_num_emb[patch_index](x_patch_num)
            x_patch_size = self.patch_size_emb[patch_index](x_patch_size)
            # B C N D
            (
                patch_num_distribution,
                patch_size_distribution,
                logi_patch_num,
                logi_patch_size,
                T_num_logi_list,
                T_size_logi_list
            ) = patch_enc(x_patch_num, x_patch_size, mask)
            patch_num_distribution_list.append(patch_num_distribution)
            patch_size_distribution_list.append(patch_size_distribution)
            recs = []
            for i in range(len(logi_patch_num)):
                logi_patch_num1 = logi_patch_num[i]
                logi_patch_size1 = logi_patch_size[i]
                patch_num_mx = self.patch_num_mixer(logi_patch_num1)
                patch_size_mx = self.patch_size_mixer(logi_patch_size1)
                patch_num_mx_list.append(patch_num_mx)
                patch_size_mx_list.append(patch_size_mx)
                rec1 = self.recons_num[patch_index](T_num_logi_list[i])
                rec2 = self.recons_size[patch_index](T_size_logi_list[i])
                if del_inter:
                    rec = rec2
                elif del_intra:
                    rec = rec1
                else:
                    rec_alpha = self.rec_alpha[patch_index]
                    rec = rec1 * rec_alpha + rec2 * (1 - rec_alpha)
                recs.append(rec)
            recs = torch.stack(recs, dim=0).mean(0)
            if not self.training:
                self.T1 = T_num_logi_list[-1]
                self.T2 = T_size_logi_list[-1]
            if rec_x is None:
                rec_x = recs
            else:
                rec_x = rec_x + recs
        rec_x = rec_x / len(self.patch_sizes)
        patch_num_distribution_list = list(_flatten(patch_num_distribution_list))
        patch_size_distribution_list = list(_flatten(patch_size_distribution_list))
        patch_num_mx_list = list(_flatten(patch_num_mx_list))
        patch_size_mx_list = list(_flatten(patch_size_mx_list))
        if self.output_attention:
            return (
                patch_num_distribution_list,
                patch_size_distribution_list,
                patch_num_mx_list,
                patch_size_mx_list,
                rec_x
            )
        else:
            return None
