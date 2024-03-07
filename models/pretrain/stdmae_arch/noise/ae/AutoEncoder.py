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

    def __init__(self, dim_X, dim_H, hidden_list=None):
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        if hidden_list is not None:
            layers = []
            for i in range(len(hidden_list) - 1):
                layers.append(nn.Linear(hidden_list[i], hidden_list[i + 1]))
                layers.append(nn.ReLU())
            self.encoder = nn.Sequential(
                nn.Linear(dim_X, hidden_list[0]),
                nn.ReLU(),
                *layers,
                nn.Linear(hidden_list[-1], dim_H)
            )
        else:
            self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        # You may want to construct a more complex decoder similar to the encoder
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, mode="enc"):
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
        if mode == "dec":
            dec = self.decoder(X)
            return dec
        elif mode == "rep":
            return self.encoder(X)
        else:
            H = self.encoder(X)
            return self.decoder(H)




