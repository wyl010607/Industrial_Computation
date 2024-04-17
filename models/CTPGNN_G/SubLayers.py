import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np


def gru_forward(input, initial_states, w_ih, w_hh, b_ih, b_hh):
    prev_h = initial_states
    bs, T, i_size = input.shape
    h_size = w_ih.shape[0] // 3
    # 对权重扩维，变成batch_size倍
    batch_w_ih = w_ih.unsqueeze(0).tile(bs, 1, 1)
    batch_w_hh = w_hh.unsqueeze(0).tile(bs, 1, 1)

    output = torch.zeros(bs, T, h_size)  # GRU网络输出序列

    for t in range(T):
        x = input[:, t, :]  # t时刻输入 [bs, i_size]
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [bs,3*hs,1]
        w_times_x = w_times_x.squeeze(-1)

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [bs,3*hs,1]
        w_times_h_prev = w_times_h_prev.squeeze(-1)

        r_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] + b_ih[:h_size] + b_hh[:h_size])  # 重置门
        z_t = torch.sigmoid(w_times_x[:, h_size:2 * h_size] + w_times_h_prev[:, h_size:2 * h_size] \
                            + b_ih[h_size:2 * h_size] + b_hh[h_size:2 * h_size])  # 更新门

        n_t = torch.tanh(w_times_x[:, 2 * h_size:3 * h_size] + b_ih[2 * h_size:3 * h_size] + \
                         r_t * (w_times_h_prev[:, 2 * h_size:3 * h_size] + b_hh[2 * h_size:3 * h_size]))  # 候选状态

        prev_h = (1 - z_t) * n_t + z_t * prev_h  # 更新隐藏状态
        output[:, t, :] = prev_h

    return output, prev_h



def laplacian(W):
    N, N = W.shape
    W = W+torch.eye(N).to(W.device)
    D = W.sum(axis=1)
    D = torch.diag(D**(-0.5))
    out = D@W@D
    return out


def matrix_fnorm(W):
    # W:(h,n,n) return (h)
    h, n, n = W.shape
    W = W**2
    norm = (W.sum(dim=1).sum(dim=1))**(0.5)
    return norm/(n**0.5)


class TPGNN(nn.Module):
    # softlap+outerwave
    def __init__(self, d_attribute, d_out, n_route, n_his, dis_mat, kt=2, n_c=10, droprate=0., temperature=1.0) -> None:
        super(TPGNN, self).__init__()
        self.droprate = droprate
        self.n_his = n_his
        self.d_attribute = d_attribute
        print(n_route, n_c)
        self.r1 = nn.Parameter(torch.randn(n_route, n_c))
        # self.r2 = nn.Parameter(torch.randn(n_route, 10))
        self.w_stack = nn.Parameter(torch.randn(kt+1, d_attribute, d_out))
        nn.init.xavier_uniform_(self.w_stack.data)
        self.reduce_stamp = nn.Linear(n_his//3*2, 1, bias=False)
        self.temp_1 = nn.Linear(d_attribute//4, kt+1)
        self.gru = nn.GRU(n_route, d_attribute//4, 2, batch_first = True)
        # self.temp_2 = nn.Linear(d_attribute//4, kt+1)
        self.temperature = temperature
        self.d_out = d_out
        self.distant_mat = dis_mat

        self.kt = kt

    def forward(self, x, stamp):
        # x:(b,n,t,k) stamp:(b,t,k)
        residual = x
        b, n, t, k = x.size()
        h0 = torch.randn(2, b, self.d_attribute//4).cuda()
        h, _, _ = self.w_stack.shape
        w_stack = self.w_stack/(matrix_fnorm(self.w_stack).reshape(h, 1, 1))
        # print(stamp.shape)

        # (b,t,k)->(b,k,1)->(b,kt+1)
        #period_emb = self.reduce_stamp(stamp.permute(0, 2, 1)).squeeze(2)
        #temp_1 = self.temp_1(period_emb)

        stamp_1, _ = self.gru(stamp.permute(0, 3, 2, 1).squeeze(1), h0)
        #stamp_1 = stamp_1[:, -1, :] 11
        #stamp_1, _ = gru_forward(stamp.permute(0, 3, 2, 1).squeeze(1), h0, self.gru.weight_ih_l0, self.gru.weight_hh_l0, self.gru.bias_ih_l0, self.gru.bias_hh_l0)

        stamp_1 = stamp_1[:, self.n_his//3:, :]

        period_emb = self.reduce_stamp(stamp_1.permute(0, 2, 1)).squeeze(2)
        temp_1 = self.temp_1(period_emb) #stamp_1
        #temp_2 = self.temp_2(period_emb)
        adj = self.distant_mat.clone()
        # adj_2 = self.cor_mat.clone()
        if self.training:
            nonzero_mask = self.distant_mat > 0
            adj[nonzero_mask] = F.dropout(adj[nonzero_mask], p=self.droprate)

        adj_1 = torch.softmax(torch.relu(
            laplacian(adj))/self.temperature, dim=0)
        adj_2 = torch.softmax(torch.relu(
            self.r1@self.r1.T)/self.temperature, dim=0)
        adj_1 = F.dropout(adj_1, p=self.droprate)
        adj_2 = F.dropout(adj_2, p=self.droprate)
        # (b,n,t,k)->(b,t,n,k)
        z = (x@w_stack[0])*(temp_1[:, 0].reshape(b, 1, 1, 1))
        z = z.permute(0, 2, 1, 3).reshape(b*t, n, -1)
        # for i in range(1, self.kt):
        #     z = adj@z + \
        #         (x@w_stack[i]*(temp[:, i].reshape(b, 1, 1, 1))
        #          ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        for i in range(1, self.kt):
            z = adj_1@z + \
                (x@w_stack[i]*(temp_1[:, i].reshape(b, 1, 1, 1))
                 ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        z_fix = (x@w_stack[0])*(temp_1[:, 0].reshape(b, 1, 1, 1))
        z_fix = z_fix.permute(0, 2, 1, 3).reshape(b*t, n, -1)
        for i in range(1, self.kt):
            z_fix = adj_2@z_fix + \
                (x@w_stack[i]*(temp_1[:, i].reshape(b, 1, 1, 1))
                 ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        z = z+z_fix

        z = z.reshape(b, t, n, self.d_out).permute(0, 2, 1, 3)

        return z
