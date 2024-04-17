import torch
import torch.nn as nn

from models.OTPGNET.fsnet import TSEncoder


class TempoEnc(nn.Module):
    def __init__(
        self,
        n_time,
        n_attr,
        normal=True
    ):
        super().__init__()

        self.time = n_time
        self.enc = nn.Embedding(n_time, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)

    def forward(self, x, start=0, t_left=None):
        length = x.shape[-2]
        if t_left == None:
            enc = self.enc(torch.arange(start, start + length).cuda())
        else:
            enc = self.enc(torch.Tensor(t_left).long().cuda())
        x = x + enc
        if self.no:
            x = self.norm(x)
        return x


class timestamp(nn.Module):
    def __init__(
        self, history_len, emb_channels
    ):
        super().__init__()
        self.time_stamp = nn.Embedding(288, emb_channels)
        # add temporal embedding and normalize
        self.tempral_enc = TempoEnc(history_len, emb_channels, True)

    def forward(self, stamp):
        time_emb = self.time_stamp(stamp)
        time_emb = self.tempral_enc(time_emb)
        return time_emb




class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input, time_emb):
        return self.encoder(input, time_emb, mask=self.mask)


class onet(nn.Module):
    def __init__(self, num_nodes, adj_mx, history_len, in_dim, out_dim, residual_channels, dilation_channels,
                 skip_channels, end_channels, emb_channels, blocks, layers, dropout, gamma, device):
        super().__init__()
        self.device = device
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        encoder = TSEncoder(device, num_nodes, supports, history_len, in_dim=in_dim, out_dim=out_dim,
                            residual_channels=residual_channels, dilation_channels=dilation_channels,
                            skip_channels=skip_channels, end_channels=end_channels, emb_channels=emb_channels,
                            blocks=blocks, layers=layers, dropout=dropout, gamma=gamma)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.stamp_emb = timestamp(history_len, emb_channels)

    def forward(self, x, stamp):
        time_emb = self.stamp_emb(stamp)
        return self.encoder(x, time_emb)

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()