from torch import nn
import torch
import torch.nn.functional as F


class STPatchEmbedding(nn.Module):
    """Patchify time series with spatial-temporal context."""

    def __init__(
        self,
        patch_size,
        in_channel,
        embed_dim,
        norm_layer,
        adj_mx,
        neighbor_simplied_num,
        adjust_adj_mx=False,
    ):
        super().__init__()
        self.num_nodes = adj_mx.shape[0]
        self.output_channel = embed_dim
        self.len_patch = patch_size  # the L
        self.input_channel = in_channel
        self.neighbor_simplied_num = neighbor_simplied_num
        self.input_embedding = nn.Conv2d(
            in_channel * self.num_nodes,
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1),
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

        # Registering adj_mx as a parameter
        self.adj_mx = nn.Parameter(torch.tensor(adj_mx, dtype=torch.float32), requires_grad=False)

        # Creating and registering a learnable parameter to adjust adj_mx
        self.adjust_adj_mx = adjust_adj_mx
        if self.adjust_adj_mx:
            self.adj_adjust_u = nn.Parameter(torch.ones_like(self.adj_mx))
            self.adj_adjust_v = nn.Parameter(torch.zeros_like(self.adj_mx))
        else:
            self.adj_adjust_u = nn.Parameter(
                torch.ones_like(self.adj_mx), requires_grad=False
            )
            self.adj_adjust_v = nn.Parameter(
                torch.zeros_like(self.adj_mx), requires_grad=False
            )

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L].
        Returns:
            torch.Tensor: Patchified time series with shape [B, N, d, P]
        """

        # Adjusting adj_mx with the learnable parameter
        adjusted_adj_mx = self.adj_adjust_u * self.adj_mx + self.adj_adjust_v
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        sampled_adj_mask = self.sample_k_neighbor_soft(adjusted_adj_mx, self.neighbor_simplied_num)

        # Sampling neighbors for each node and concatenating their features
        expanded_history = long_term_history.expand(
            -1, -1, num_nodes, -1
        )  # Expand to [B, N, N, P*L]
        neighbors_data = sampled_adj_mask.unsqueeze(0).unsqueeze(-1) * expanded_history
        #neighbors_data = self.neighbor_aggregation(neighbors_data.transpose(-1,-2)).squeeze(-1)
        neighbors_data = neighbors_data.reshape(
            batch_size * num_nodes,
            num_nodes,
            len_time_series,
            1,
        )

        # Convolution and normalization
        output = self.input_embedding(neighbors_data)
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch

        return output

    def sample_k_neighbor_soft(self, adj_mx, k, tau=1.0):
        """
        Samples k neighbors for each node using a soft differentiable mask based on Gumbel-Softmax and top-k selection.
        Args:
            adj_mx (torch.Tensor): The adjusted adjacency matrix of the graph. Shape: [N, N]
            k (int): The number of neighbors to sample.
            tau (float): Temperature parameter for Gumbel-Softmax distribution. Lower values make the output more discrete.
        Returns:
            torch.Tensor: A soft mask matrix. Shape: [N, N]
        """
        N = adj_mx.shape[0]
        adjusted_adj_mx = torch.relu(adj_mx) + 1e-8
        logits = torch.log(adjusted_adj_mx)

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(adj_mx) + 1e-8))
        gumbel_logits = (logits + gumbel_noise) / tau
        soft_masks = F.softmax(gumbel_logits, dim=1)

        topk_values, topk_indices = torch.topk(soft_masks, k, dim=1)
        mask = torch.zeros_like(soft_masks).scatter_(1, topk_indices, 1)
        enhanced_soft_masks = soft_masks * mask

        return enhanced_soft_masks