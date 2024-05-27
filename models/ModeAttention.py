from torch import nn
import torch
import numpy as np
from models.utils import ACTIVATION_MAP, RNN_MAP
import torch.nn.functional as F
# class Attention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Attention, self).__init__()
#         self.fc = nn.Linear(input_size, hidden_size)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, input_data):
#         attention_weights = self.fc(input_data)
#         attention_weights = self.softmax(attention_weights)
#         return attention_weights*input_data


class ModeAttention(nn.Module):
    def __init__(self, cell, sequence_len, feature_num, hidden_dim,
                 fc_layer_dim, rnn_num_layers, output_dim, fc_activation,
                 attention_order, feature_head_num=None, sequence_head_num=None,
                 fc_dropout=0, rnn_dropout=0, bidirectional=False, return_attention_weights=False,emd_num=3,hidden_num=1,dropout_prob=0.5,mlp_num=10):
        super().__init__()
        assert cell in ['rnn', 'lstm', 'gru']
        assert fc_activation in ['tanh', 'gelu', 'relu']
        assert isinstance(attention_order, list)
        self.feature_num = feature_num
        self.emd_num=emd_num
        self.hidden_num=hidden_num
        self.mlp_num=mlp_num;
        self.dropout_prob = dropout_prob

        # #1
        # self.fc1 = nn.ModuleList([
        #     nn.Linear(self.emd_num, 10) for _ in range(self.mlp_num)
        # ])
        # self.fc2 = nn.ModuleList([
        #     nn.Linear(10, 10) for _ in range(self.mlp_num)
        # ])
        # self.fc3 = nn.ModuleList([
        #     nn.Linear(10, self.emd_num) for _ in range(self.mlp_num)
        # ])
        # self.fc4=nn.Linear(self.mlp_num, 10)
        # self.fc5=nn.Linear(10, 10)
        # self.fc6=nn.Linear(10,1)
        # self.dropout1 = nn.Dropout(p=self.dropout_prob)

        # #2
        # self.fc1 = nn.ModuleList([
        #     nn.Linear(self.emd_num, 10) for _ in range(self.mlp_num)
        # ])
        # self.fc2 = nn.ModuleList([
        #     nn.Linear(10, 10) for _ in range(self.mlp_num)
        # ])
        # self.fc3 = nn.ModuleList([
        #     nn.Linear(10, 1) for _ in range(self.mlp_num)
        # ])
        # self.fc4=nn.Linear(self.mlp_num, 10)
        # self.fc5=nn.Linear(10, 10)
        # self.fc6=nn.Linear(10,1)
        # self.dropout1 = nn.Dropout(p=self.dropout_prob)

        #3
        self.fc1=nn.Linear(self.emd_num,6)
        self.fc2=nn.Linear(6,6)
        self.fc3=nn.Linear(6,self.emd_num)
        self.dropout1 = nn.Dropout(p=self.dropout_prob)


        

        self.sequence_len = sequence_len
        self.cell = cell
        self.rnn_hidden_size = hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout
        self.fc_activation = fc_activation

        self.output_dim = output_dim
        self.rnn_dropout = rnn_dropout

        self.feature_head_num = feature_head_num
        self.sequence_head_num = sequence_head_num

        self.return_attention_weights = return_attention_weights

        self.attention_layers = nn.ModuleList()

        
        self.attention_order = []
        for attn_type in attention_order:
            if attn_type == 'feature' and self.feature_head_num > 0:
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.sequence_len,
                        num_heads=self.feature_head_num
                    )
                )
                self.attention_order.append('feature')
            if attn_type == 'sequence' and self.sequence_head_num > 0:
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.feature_num,
                        num_heads=self.sequence_head_num
                    )
                )
                self.attention_order.append('sequence')

        self.rnn = RNN_MAP[self.cell](
            input_size=self.feature_num,
            hidden_size=self.rnn_hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            num_layers=self.rnn_num_layers,
            dropout=self.rnn_dropout
        )
        linear_in_size = self.rnn_hidden_size
        if self.bidirectional:
            linear_in_size *= 2
        self.linear = nn.Sequential(
            nn.Linear(linear_in_size, self.fc_layer_dim),
            ACTIVATION_MAP[self.fc_activation](),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, output_dim),
        )

        # nn.init.kaiming_normal_(self.linear[0].weight, mode='fan_in')
        # nn.init.kaiming_normal_(self.linear[3].weight, mode='fan_in')
        if self.cell == 'lstm':
            # hidden_size = self.rnn_hidden_size
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
            # nn.init.zeros_(self.rnn.bias_ih_l0)
            # nn.init.ones_(self.rnn.bias_ih_l0[hidden_size:hidden_size * 2])

    def forward(self, x):
        # #1
        # n=x.shape[0]
        # x_mean = torch.mean(x, dim=1, keepdim=True)
        # x_mean = x_mean.permute(0,2,1)
        # x_mean=torch.squeeze(x_mean, axis=2)
        # x_mean=torch.reshape(x_mean,(n,self.feature_num,self.emd_num))
        # x = torch.reshape(x,(n,self.sequence_len,self.emd_num,self.feature_num))
        # x = x.permute(0,3,1,2)
        # x_list=[]
        # for i in range(self.mlp_num):
        #     fc_output = F.relu(self.fc1[i](x_mean))
        #     fc_output = F.relu(self.fc2[i](fc_output))
        #     fc_output = F.softmax(self.fc3[i](fc_output),dim=-1)
        #     fc_output=fc_output.unsqueeze(-1);
        #     x_i=torch.matmul(x,fc_output).squeeze(dim=-1)
        #     x_list.append(x_i)
        # x_cat = torch.cat(x_list, dim=1)
        # x_cat=x_cat.permute(0,2,1)
        # x=torch.reshape(x_cat,(n,self.sequence_len,self.feature_num,self.mlp_num))
        # x=F.relu(self.fc4(x))
        # x=F.relu(self.fc5(x))
        # x=F.relu(self.fc6(x)).squeeze(dim=-1)
        # x=self.dropout1(x)

        # #2
        # n=x.shape[0]
        # x = torch.reshape(x,(n,self.sequence_len,self.feature_num,self.emd_num))
        # x_list=[]
        # for k in range(self.mlp_num):
        #   for i in range(self.feature_num):
        #       fc_output = F.relu(self.fc1[k](x[:,:,i,:]))
        #       fc_output = F.relu(self.fc2[k](fc_output))
        #       fc_output = F.relu(self.fc3[k](fc_output))
        #       fc_output=fc_output.unsqueeze(-1);
        #       x_list.append(fc_output)
        # x_cat = torch.cat(x_list, dim=2)
        # x=torch.reshape(x_cat,(n,self.sequence_len,self.feature_num,self.mlp_num))
        # x=F.relu(self.fc4(x))
        # x=F.relu(self.fc5(x))
        # x=F.relu(self.fc6(x)).squeeze(dim=-1)
        # x=self.dropout1(x)


        #3
        n=x.shape[0]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean = x_mean.permute(0,2,1).squeeze(dim=-1)
        x_mean = torch.reshape(x_mean,(n,self.feature_num,self.emd_num))
        x_mean=F.relu(self.fc1(x_mean))
        x_mean=F.relu(self.fc2(x_mean))
        x_mean=F.relu(self.fc3(x_mean)).unsqueeze(dim=-1)
        # x_mean=self.dropout1(x_mean)
        x=torch.reshape(x,(n,self.sequence_len,self.feature_num,self.emd_num))
        x=x.permute(0,2,1,3)
        x=torch.matmul(x,x_mean).squeeze(dim=-1)
        x=x.permute(0,2,1)

        # #4
        # n=x.shape[0]
        # x = torch.reshape(x,(n,self.sequence_len,self.feature_num,self.emd_num))
        # x=torch.mean(x, dim=-1, keepdim=True).squeeze(dim=-1)





        # print(x_mean)
        # x=torch.reshape(x,(n,self.sequence_len,self.feature_num,self.emd_num))
        
        # x_mean = F.relu(self.fc1(x_mean))
        # x_mean = self.dropout1(x_mean)
        
        # # x_mean = F.relu(self.fc2(x_mean))
        # # x_mean = self.dropout2(x_mean)
        # x_mean = F.softmax(self.fc3(x_mean),dim=-1)
        # x_mean=x_mean.unsqueeze(-1);
        # x = torch.reshape(x,(n,self.sequence_len,self.emd_num,self.feature_num))
        # x = x.permute(0,3,1,2) 
        # x=torch.matmul(x,x_mean).squeeze(dim=-1)
        # x = x.permute(0,2,1) 

        # x_cat = torch.cat(x_list, dim=2)
        # x_cat=torch.squeeze(x_cat,dim=-1)
        # x=torch.reshape(x_cat,(n,self.sequence_len,self.feature_num,self.mlp_num))
        # x=F.relu(self.fc4(x))
        # x=F.relu(self.fc5(x))
        # x=F.relu(self.fc6(x)).squeeze(dim=-1)
        # print(x.shape)

        # x_mean=torch.reshape(x_mean,(n,self.feature_num,self.hidden_num*self.mlp_num))
        # x_mean=F.relu(self.fc2(x_mean))
        # x_mean=torch.squeeze(x_mean, axis=2)
        # x_mean = F.softmax(self.fc3(x_mean),dim=-1)
        
        # x_mean_expanded = x_mean.unsqueeze(1).unsqueeze(-1)
        # # x_mean_expanded = x_mean_expanded.permute(0,2,1,3)
        # x = torch.reshape(x,(n,self.sequence_len,self.emd_num,self.feature_num))
        # x = x.permute(0,3,2,1) 
        # x = torch.sum(x * x_mean_expanded, dim=2)
        # x = x.permute(0,2,1)
        
        feature_weights = None
        sequence_weights = None
        for i, module in enumerate(self.attention_layers):
            attn_type = self.attention_order[i]
            if attn_type == 'feature':
                a_in = x.permute(2, 0, 1)
                print('feature',a_in.shape)
                x, feature_weights = module(a_in, a_in, a_in)
                x = x.permute(1, 2, 0)
                print('feature',x.shape)
            if attn_type == 'sequence':
                a_in = x.permute(1, 0, 2)
                x, sequence_weights = module(a_in, a_in, a_in)
                x = x.permute(1, 0, 2)

        # LSTM/GRU
        # print(x.shape)
        x, _ = self.rnn(x)
        # print('lstm',x.shape)

        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        # print(x.shape)
        x = self.linear(x)
        # print('linear',x.shape)
        if self.return_attention_weights:
            return x, feature_weights, sequence_weights
        else:
            return x

    