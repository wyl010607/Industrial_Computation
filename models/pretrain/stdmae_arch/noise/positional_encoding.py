import torch
from torch import nn
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    PositionalEncoding3D,
    Summer,
)


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_feat):
        super().__init__()
        self.tp_enc_2d = PositionalEncoding2D(num_feat)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data += self.tp_enc_2d(input_data)
        return input_data, self.tp_enc_2d(input_data)
