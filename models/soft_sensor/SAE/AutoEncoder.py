import torch
from torch import nn


class AutoEncoder(nn.Module):
    '''
    AutoEncoder

    Parameters
    ----------
    dim_X : int
        The dimension of the input data.
    dim_H : int
        The dimension of the hidden layer.
    '''
    def __init__(self, dim_X, dim_H):
        """
        Parameters
        ----------
        dim_X : int
            The dimension of the input data.
        dim_H : int
            The dimension of the hidden layer.
        """
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        self.act = torch.sigmoid

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, rep=False):
        """
        Parameters
        ----------
        X : torch.Tensor
            The input data. shape as [batch_size, dim_X]
        rep : bool
            Whether to return the representation of the input data. Default: False

        Returns
        -------

        """
        H = self.act(self.encoder(X))
        if rep is False:
            return self.act(self.decoder(H))
        else:
            return H
