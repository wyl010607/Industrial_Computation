from torch import nn
import torch


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
        self.output_channel = embed_dim
        self.len_patch = patch_size  # the L
        self.input_channel = in_channel
        self.neighbor_simplied_num = neighbor_simplied_num
        self.input_embedding = nn.Conv2d(
            in_channel * (neighbor_simplied_num + 1),
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1),
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

        # Registering adj_mx as a parameter
        self.adj_mx = nn.Parameter(torch.tensor(adj_mx), requires_grad=False)

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
        sampled_adj = self.sample_k_neighbor(adjusted_adj_mx, self.neighbor_simplied_num)

        # Sampling neighbors for each node and concatenating their features
        neighbors_data = torch.zeros(
            batch_size,
            num_nodes,
            self.input_channel * (self.neighbor_simplied_num + 1),
            len_time_series,
            device=long_term_history.device,
        )
        for i in range(num_nodes):
            neighbors = sampled_adj[i]
            neighbor_data = long_term_history[:, neighbors].view(
                batch_size, -1, len_time_series
            )
            # Concatenating the features of the node and its neighbors
            neighbors_data[:, i, :, :] = torch.cat(
                (long_term_history[:, i], neighbor_data), dim=1
            )

        neighbors_data = neighbors_data.reshape(
            batch_size * num_nodes,
            self.input_channel * (self.neighbor_simplied_num + 1),
            len_time_series,
            1,
        )

        # Convolution and normalization
        output = self.input_embedding(neighbors_data)
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch

        return output

    def sample_k_neighbor(self, adj_mx, k):
        """
        Samples k neighbors for each node based on adjusted adjacency matrix probabilities.
        Args:
            adj_mx (torch.Tensor): The adjusted adjacency matrix of the graph. shape: [N, N]
            k (int): The number of neighbors to sample.
        Returns:
            torch.Tensor: The sampled neighbors indices. shape: [N, k]
        """
        N = adj_mx.shape[0]
        sampled_neighbors = torch.zeros((N, k), dtype=torch.int64)

        for i in range(N):
            probs = adj_mx[i] / torch.sum(adj_mx[i])
            neighbors = torch.multinomial(probs, k, replacement=True)
            sampled_neighbors[i] = neighbors

        return sampled_neighbors
