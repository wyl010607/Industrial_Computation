import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    scaled_laplacian = scaled_laplacian(diag_matrix)
    res = [np.identity(scaled_laplacian.shape[0]), scaled_laplacian]
    if K >= 2:
        for i in range(2, K + 1):
            res.append(2 * scaled_laplacian * res[i - 1] - res[i - 2])
    return res


class Cheb_conv(nn.Module):
    """
    Take chebyshev ploynomials as kernal for graph conv.
    """

    def __init__(self, chebyshev_ploynomials, K, in_channel, out_channel):
        super(Cheb_conv, self).__init__()
        self.K = K
        self.chebyshev_ploynomials = chebyshev_ploynomials
        self.device = chebyshev_ploynomials[0].device
        self.thetaList = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(in_channel, out_channel).to(self.device))
                for _ in range(K)
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Calculate spectral graph convolution.

        Parameters
        ----------
        x : (batch, nodes_num, features(in_channel), seq_len)

        Returns
        -------
        res : the result of spectral graph convolution.
        """
        batch_size, nodes_num, in_channel, seq_len = x.shape
        # (batch, seq_len, nodes_num, in_channel)
        x = x.permute(0, 3, 1, 2)
        output = torch.zeros(batch_size, nodes_num, self.out_channels, seq_len).to(
            self.DEVICE
        )
        for i in range(self.K):
            cheb = self.chebyshev_ploynomials[i]
            # (batch, seq_len, nodes_num, in_channel)
            cash = cheb.matmul(x)
            # (batch, seq_len, nodes_num, out_channel)
            cash = self.cash.matmul(self.thetaList[i])
            # (batch, nodes_num, out_channel, seq_len)
            cash = cash.permute(0, 2, 3, 1)
            output += cash

        return self.relu(output)


class GluLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.conv2D(
            in_chanells=input_size,
            out_chanells=output_size,
            kernal_size=3,
            dilation=2,
            padding=2,
        )
        self.fc2 = nn.conv2D(
            in_chanells=input_size,
            out_chanells=output_size,
            kernal_size=3,
            dilation=2,
            padding=2,
        )
        self.glu = nn.GLU(dim=1)

    def forward(self, o, x, y):
        a = self.fc1(x) + o
        b = self.fc2(y)
        return self.glu(torch.cat((a, b), dim=1))


class BIAS(nn.Module):
    def __init__(self, history_len, forecast_len, num_nodes, channel, K, loop_num):
        super(BIAS, self).__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_nodes = num_nodes
        self.channel = channel
        self.K = 2
        self.loop_num = 3
        self.fc = nn.con2D(channel, forecast_len, 3)

    def forward(self, x):
        o = x
        # (batch, seq_len, nodes_num, in_channel)
        o = o.permute(0, 3, 1, 2)
        # (batch, in_channel, seq_len, nodes_num)
        for i in range(self.loop_num):
            # temporal layer
            if o.shape[1] & 1:
                o_reshape = torch.from_numpy(
                    np.pad(o, [(0, 0), (0, 1), (0, 0), (0, 0)], mode='constant')
                )
            else:
                o_reshape = o
            half = int(o_reshape.shape[1] / 2)
            split = torch.split(o_reshape, half, dim=1)
            x_u = split[0]
            x_v = split[1]
            m = GluLayer(half, half)
            target_size = (half, o_reshape.shape[-1])
            c = F.interpolate(o_reshape, size=target_size, mode='nearest')
            x_glu = m(c, x_u, x_v)
            # spatial layer
            adj = np.load('./data/jl_0701_0801_corr_adj_mx.npy')
            cheb = chebyshev_ploynomials(adj, self.K)
            sgc = Cheb_conv(cheb, self.K, half, self.channel)
            s = sgc(x_glu)
            o = o + s
        bias = self.fc(o)
        bias = bias.permute(0, 2, 3, 1)
        return bias
