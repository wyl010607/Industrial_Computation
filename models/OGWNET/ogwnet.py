import torch
import torch.nn as nn

from models.OGWNET.fsnet import TSEncoder



class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)


class ogwnet(nn.Module):
    def __init__(self, num_nodes, adj_mx, history_len, in_dim, out_dim, residual_channels, dilation_channels,
                 skip_channels, end_channels, emb_channels, blocks, layers, dropout, gamma, device):
        super().__init__()
        self.device = device
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        encoder = TSEncoder(device, num_nodes, supports, in_dim=in_dim, out_dim=out_dim,
                            residual_channels=residual_channels, dilation_channels=dilation_channels,
                            skip_channels=skip_channels, end_channels=end_channels,
                            blocks=blocks, layers=layers, dropout=dropout, gamma=gamma)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)

    def forward(self, x, stamp):
        return self.encoder(x)

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()