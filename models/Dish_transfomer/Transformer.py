import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, forecast_len,output_attention,enc_in,dec_in,d_model,embed,freq,dropout,factor,n_heads,d_ff,e_layers,activation,d_layers,c_out):
        super(Transformer, self).__init__()
        self.forecast_len = forecast_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.embed_type = 3
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.activation = activation
        self.c_out=c_out
        #print(self.forecast_len)
        # Embedding
        if self.embed_type == 0:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                            self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        elif self.embed_type == 1:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)

        elif self.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
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
                        FullAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #x_enc=x_enc.squeeze(-1)
        #x_dec=x_dec.squeeze(-1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.forecast_len:, :], attns
        else:
            return dec_out[:, -self.forecast_len:, :]  # [B, L, D]
            #return dec_out[:, -self.forecast_len:, :].unsqueeze(-1)