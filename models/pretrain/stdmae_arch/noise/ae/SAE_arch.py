import torch
from torch import nn
from .AutoEncoder import AutoEncoder


class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_list):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        hidden_dim_list : list[int]
            The dimension of the hidden layers.
        """

        super(StackedAutoEncoder, self).__init__()
        self.SAE = nn.ModuleList()
        self.SAE.append(AutoEncoder(input_dim, hidden_dim_list[0]))
        for i in range(1, len(hidden_dim_list)):
            self.SAE.append(AutoEncoder(hidden_dim_list[i-1], hidden_dim_list[i]))
        self.AElength = len(self.SAE)
        self.decoder = nn.Linear(hidden_dim_list[-1], output_dim, bias=True)

    def forward(self, X, is_pretrain=False, pretrain_layer_numero=0):
        """
        Parameters
        ----------
        X : torch.Tensor
            The input data. shape as [batch_size, dim_X]
        is_pretrain : bool
            Whether to pretrain the model. Default: False
        pretrain_layer_numero : int
            The layer number to pretrain. Default: 0

        Returns
        -------

        """
        input = X
        if is_pretrain is True:
            if pretrain_layer_numero == 0:
                return input, self.SAE[pretrain_layer_numero](input)
            else:
                hidden = input
                for i in range(pretrain_layer_numero):
                    hidden = self.SAE[i](hidden, rep=True)
                out = self.SAE[pretrain_layer_numero](hidden)
                return hidden, out
        else: # fine-tuning
            hidden = input
            for i in range(self.AElength):
                hidden = self.SAE[i](hidden, rep=True)
            out = torch.sigmoid(self.decoder(hidden))
            return out