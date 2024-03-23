import torch
import torch.nn as nn
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .SelfAttention_Family import DSAttention, AttentionLayer
from .Embed import DataEmbedding


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class NS(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self,output_attention,enc_in,d_model,dec_in,history_len,forecast_len,label_len,embed,freq,dropout,c_out,p_hidden_layers,e_layers,d_layers,d_ff,factor,n_heads,activation,p_hidden_dims,*args,**kwags):
        super(NS, self).__init__()
        #self.forecast_len = forecast_len
        self.forecast_len = 1
        self.history_len = history_len
        self.label_len = label_len
        self.output_attention = output_attention
        self.enc_in=enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.c_out = c_out
        self.p_hidden_layers = p_hidden_layers
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff=d_ff
        self.factor=factor
        self.n_heads=n_heads
        self.activation=activation
        self.p_hidden_dims=p_hidden_dims
        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        DSAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

        self.tau_learner   = Projector(enc_in=self.enc_in, seq_len=self.history_len, hidden_dims=self.p_hidden_dims, hidden_layers=self.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.enc_in, seq_len=self.history_len, hidden_dims=self.p_hidden_dims, hidden_layers=self.p_hidden_layers, output_dim=self.history_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print(x_enc.shape)
        #print(x_dec.shape)
        x_enc=x_enc.squeeze(-1)
        x_dec=x_dec.squeeze(-1)
        x_raw = x_enc.clone().detach()
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc
        #print(x_enc.shape)
        x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.forecast_len:, :])], dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
        if self.output_attention:
            return dec_out[:, -self.forecast_len:, :], attns
        else:
            return dec_out[:, -self.forecast_len:, :].unsqueeze(-1)  # [B, L, D]
