import torch
from torch import nn

from .mask import STDMask
from .graphwavenet import GraphWaveNet


class STDMAE(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""

    def __init__(self, mask_args, backend_args, **kwargs):
        super().__init__()
        # iniitalize 
        self.tmae = STDMask(**mask_args)
        self.smae = STDMask(**mask_args)
        self.num_nodes = kwargs["num_nodes"]
        self.adj_mx = kwargs["adj_mx"]
        self.backend = GraphWaveNet(**backend_args, num_nodes=self.num_nodes, adj_mx=self.adj_mx)


    def load_pretrained_model(self, tmae_model_save_path, smae_model_save_path):
        """Load pre-trained model"""

        # load parameters
        self.tmae.load_state_dict(torch.load(tmae_model_save_path))
        self.smae.load_state_dict(torch.load(smae_model_save_path))

        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        # reshape
        short_term_history = history_data     # [B, L, N, 1]

        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = self.tmae(long_history_data[..., [0]])
        hidden_states_s = self.smae(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        
        # enhance
        out_len=1
        hidden_states = hidden_states[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)

        return y_hat

