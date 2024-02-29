import torch
import torch.nn as nn

from models.FSNET.fsnet import TSEncoder


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]


class net(nn.Module):
    def __init__(self, enc_in, c_out, forecast_len, device = "cuda:0"):
        super().__init__()
        self.device = device
        encoder = TSEncoder(input_dims=enc_in + 7,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = c_out * forecast_len

        # self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim).to(self.device)

    def forward(self, x):
        rep = self.encoder(x)
        y = self.regressor(rep)
        return y

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()