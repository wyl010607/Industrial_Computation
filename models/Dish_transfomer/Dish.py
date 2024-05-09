import torch
import torch.nn as nn
from .DishTS import DishTS
from .Transformer import Transformer
class Dish(nn.Module):
    def __init__(self,output_attention,enc_in,d_model,dec_in,history_len,forecast_len,label_len,embed,freq,dropout,c_out,p_hidden_layers,e_layers,d_layers,d_ff,factor,n_heads,activation,p_hidden_dims,*args,**kwags):
        super().__init__()
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
        self.fm = Transformer(forecast_len=self.forecast_len,output_attention=self.output_attention,enc_in=self.enc_in,dec_in=self.dec_in,d_model=self.d_model,embed=self.embed,freq=self.freq,dropout=self.dropout,factor=self.factor,n_heads=self.n_heads,d_ff=self.d_ff,e_layers=self.e_layers,activation=self.activation,d_layers=self.d_layers,c_out=self.c_out)
        self.nm = DishTS(enc_in=self.enc_in,history_len=self.history_len)
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        batch_x=batch_x.squeeze(-1)
        dec_inp=dec_inp.squeeze(-1)
        batch_x, dec_inp = self.nm(batch_x, 'forward', dec_inp)

        forecast = self.fm(batch_x, None, dec_inp, None)

        forecast = self.nm(forecast, 'inverse')

        #return forecast
        return forecast.unsqueeze(-1)