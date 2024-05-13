import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN import GCN


def get_d_matrix(adj_matrix):
    """
    Get degree matrix based on adjacency matrix.

    Parameters
    ----------
    adj_matrix : adjacency matrix.

    Returns
    -------
    d_matrix : degree matrix.

    """
    d_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
    for i in range(adj_matrix.shape[0]):
        d_matrix[i, i] = np.sum(adj_matrix[i, :])
    return d_matrix


def get_laplace_matrix(adj, d):
    """
    get the normalized graph Laplacian L = I - D^(-1/2)AD^(-1/2).

    Parameters
    ----------
    adj : adjacency matrix.
    d : degree matrix.

    Returns
    -------
    res : normalized graph Laplacian.

    """
    d_turn = np.zeros((adj.shape[0], adj.shape[0]))
    for i in range(adj.shape[0]):
        d_turn[i, i] = np.power(d[i, i], -0.5)  # calculate D^(-1/2)
    res = np.eye(adj.shape[0]) - d_turn @ adj @ d_turn
    return res


def scaled_laplacian(L):
    """
    Get scaled laplacian for chebyshev ploynomials.

    Parameters
    ----------
    L : the diagonal matrix composed of eigenvalues of normalized graph Laplacian.

    Returns
    -------
    res : scaled laplacian for chebyshev ploynomials.

    """
    le = 2  # assume the largest eigenvalue is 2
    res = (2 * L) / le - np.identity(L.shape[0])
    return res


def chebyshev_ploynomials(adj_matrix, K):
    """
    Get chebyshev ploynomials.

    Parameters
    ----------
    adj_matrix : adjacency matrix.
    K : the graph convolution kernel size.

    Returns
    -------
    res : chebyshev ploynomials list.
    """
    d_matrix = get_d_matrix(adj_matrix)
    L = get_laplace_matrix(adj_matrix, d_matrix)
    eigvals, eigvecs = np.linalg.eig(L)
    diag_matrix = np.diag(eigvals)
    scaled = scaled_laplacian(diag_matrix)
    res = [np.identity(scaled.shape[0]), scaled]
    if K >= 2:
        for i in range(2, K + 1):
            res.append(2 * scaled * res[i - 1] - res[i - 2])
    return res


class ChebConv(torch.nn.Module):
    """
    Graph Convolution with Chebyshev polynominals.

    Args:
        - input_feature: Dimension of input features.
        - out_feature: Dimension of output features.
        - adj_mx: Adjacent matrix with shape :math:`(K, num\_nodes, num\_nodes)` followed by
          Kth Chebyshev polynominals, where :math:`T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x-2)` with
          :math:`T_0(x)=1, T_1(x) = x`.

    Shape:
        - Input:
            x: :math:`(batch\_size, c_in, num\_nodes, f_in)`.
            spatial_att: :math:`(batch\_size, num\_nodes, num\_nodes)`
        - Output:
            :math:`(batch_size, c_in, num_\nodes, f_out).
    """

    def __init__(self, input_feature, out_feature, adj_mx):
        super(ChebConv, self).__init__()

        self.adj_mx = adj_mx
        self.w = torch.nn.Linear(
            adj_mx.size(0) * input_feature, out_feature, bias=False
        ).cuda()

    def forward(self, x):
        b, c_in, num_nodes, _ = x.size()
        outputs = []
        adj = self.adj_mx.unsqueeze(dim=0)
        for i in range(c_in):
            x1 = x[:, i].unsqueeze(dim=1)
            y = torch.matmul(adj, x1).transpose(1, 2).reshape(b, num_nodes, -1)
            y = torch.relu(self.w(y))
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class GluLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            dilation=2,
            padding=2,
        )
        self.fc2 = nn.Conv2d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            dilation=2,
            padding=2,
        )
        self.glu = nn.GLU(dim=1)

    def forward(self, x, y):
        a = self.fc1(x)
        b = self.fc2(y)
        return self.glu(torch.cat((a, b), dim=1))
    
class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        # Discussion on odd and even numbers:
        self.P[:, :, 0::2] = torch.sin(X)
        if num_hiddens & 1:
            self.P[:, :, 1::2] = torch.cos(X[:, : num_hiddens // 2])
        else:
            self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class BIAS(nn.Module):
    def __init__(
        self,
        batch_size,
        history_len,
        num_nodes,
        channel,
        forecast_len,
        K,
        loop_num,
        adj,
    ):
        super(BIAS, self).__init__()
        self.batch_size = batch_size
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_nodes = num_nodes
        self.channel = channel
        self.K = 2
        self.loop_num = 1
        self.adj = adj
        self.fc = nn.Conv2d(
            in_channels=channel, out_channels=forecast_len, kernel_size=(1, 64)
        )
        self.gcn_conv = torch.nn.Conv2d(
            in_channels=self.history_len,
            out_channels=self.forecast_len,
            kernel_size=(1, 64),
        ).cuda()
        self.cheb_conv = ChebConv(1, 64, self.adj)

    def forward(self, x):
        o = x.permute(3, 1, 0, 2).reshape(self.channel, self.history_len, -1)
        pos_encoding = PositionalEncoding(self.num_nodes * self.batch_size, 0).eval()
        X = (
            pos_encoding(o)
            .reshape(self.channel, self.history_len, self.batch_size, self.num_nodes)
            .permute(2, 1, 3, 0)
        )
        o = X
        # (batch, seq_len, nodes_num, in_channel)
        for i in range(self.loop_num):
            # temporal layer
            o_reshape = x
            half = int(o_reshape.shape[1] / 2)
            split = torch.split(o_reshape, half, dim=1)
            x_u = split[0]
            x_v = split[1]
            m = GluLayer(half, self.forecast_len).cuda()
            x_glu = m(x_u, x_v)
            # spatial layer
            cheb = self.cheb_conv(x)
            sgc = self.gcn_conv(cheb)
        bias = x_glu + sgc
        return bias
