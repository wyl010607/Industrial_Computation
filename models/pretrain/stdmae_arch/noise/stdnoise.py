import math

import torch
from torch import nn
from .patch import PatchEmbedding
from .STpatch import STPatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from .ae import AutoEncoder


def unshuffle(shuffled_tokens):
    dic = {}
    for (
        k,
        v,
    ) in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class STDNoise(nn.Module):
    def __init__(
        self,
        patch_size,
        in_channel,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout,
        low_rank,
        noise_type,
        noise_intensity,
        encoder_depth,
        decoder_depth,
        patch_method="patch",
        adj_mx=None,
        spatial=False,
        mode="pre-train",
        low_rank_method="pca",
        ae_low_rank_hidden_list=None,
    ):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.low_rank = low_rank
        self.low_rank_method = low_rank_method
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial
        self.selected_feature = 0
        self.patch_method = patch_method
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None
        # encoder specifics
        # # patchify & embedding
        self.adj_mx = adj_mx
        if patch_method == "patch":
            self.patch_embedding = PatchEmbedding(
                patch_size, in_channel, embed_dim, norm_layer=None
            )
        elif patch_method == "STpatch":
            self.patch_embedding = STPatchEmbedding(
                patch_size,
                in_channel,
                embed_dim,
                norm_layer=None,
                adj_mx=adj_mx,
                neighbor_simplied_num=3,
                adjust_adj_mx=False,
            )

        # positional encoding to device
        self.positional_encoding = PositionalEncoding(embed_dim)
        # encoder
        self.encoder = TransformerLayers(
            embed_dim, encoder_depth, mlp_ratio, num_heads, dropout
        )

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # decoder
        self.decoder = TransformerLayers(
            embed_dim, decoder_depth, mlp_ratio, num_heads, dropout
        )

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)

        if low_rank_method == "ae":
            dim_x = embed_dim
            self.ae = AutoEncoder(
                dim_X=dim_x,
                dim_H=low_rank,
                hidden_list=ae_low_rank_hidden_list,
            )

    def pca_low_rank(self, input, q, axis=0):
        """PCA low rank approximation., use torch.pca_lowrank

        Args:
            input (torch.Tensor): input tensor.
            q (int): rank.
            axis (int): axis.

        Returns:
            torch.Tensor: low rank approximation.
        """
        U, S, V = torch.pca_lowrank(input, q=q, center=True, niter=3)
        return (
            torch.matmul(U, torch.diag(S)[:, :q]),
            V,
            input.mean(dim=axis, keepdim=True),
        )

    def inv_pca_low_rank(self, input, V, mean, q, axis=0):
        """inverse PCA low rank approximation.

        Args:
            input (torch.Tensor): input tensor.
            V (torch.Tensor): V matrix.
            mean (torch.Tensor): mean.
            q (int): rank.
            axis (int): axis.

        Returns:
            torch.Tensor: reconstructed tensor.
        """
        return torch.matmul(input, V[:, :q].T) + mean

    def batch_pca(self, input_tensor, k):
        # 中心化
        mean = input_tensor.mean(dim=1, keepdim=True)
        centered_tensor = input_tensor - mean

        ## 对张量进行批处理SVD
        # U, S, Vh = torch.linalg.svd(centered_tensor, full_matrices=False)
        ## 选择前k个主成分
        # U_k = U[:, :, :k]
        # S_k = S[:, :k]
        # Vh_k = Vh[:, :k, :]

        U_k, S_k, Vh_k = [], [], []
        for i in range(input_tensor.size(0)):
            U, S, Vh = torch.pca_lowrank(centered_tensor[i], q=k, center=False)
            U_k.append(U)
            S_k.append(S)
            Vh_k.append(Vh.T)
        U_k = torch.stack(U_k)
        S_k = torch.stack(S_k)
        Vh_k = torch.stack(Vh_k)

        # 进行PCA降维
        reduced_tensor = torch.bmm(U_k, torch.diag_embed(S_k))

        return reduced_tensor, U_k, S_k, Vh_k, mean

    def batch_inv_pca(self, reduced_tensor, U_k, S_k, Vh_k, mean):
        # 进行逆PCA变换
        reconstructed_tensor = torch.bmm(reduced_tensor, Vh_k) + mean

        return reconstructed_tensor

    def add_noise(self, input_tensor, noise_type, noise_intensity):
        if isinstance(noise_type, str):
            noise_type = [noise_type]

        for nt in noise_type:
            if nt == "gaussian":
                mean = noise_intensity[nt].get("mean", 0)
                std = noise_intensity[nt].get("std", 1)
                noise = torch.randn_like(input_tensor) * std + mean
                input_tensor += noise
            elif nt == "uniform":
                a = noise_intensity[nt].get("a", -1)
                b = noise_intensity[nt].get("b", 1)
                noise = (b - a) * torch.rand_like(input_tensor) + a
                input_tensor += noise
            elif nt == "salt_pepper":
                prob = noise_intensity[nt].get("prob", 0.05)
                salt_val = noise_intensity[nt].get("salt_val", 1)
                pepper_val = noise_intensity[nt].get("pepper_val", 0)
                mask = torch.rand_like(input_tensor) < prob
                salt_pepper_noise = torch.rand_like(input_tensor) < 0.5
                salt_pepper_noise = (
                    salt_pepper_noise.float() * salt_val
                    + (1 - salt_pepper_noise.float()) * pepper_val
                )
                input_tensor[mask] = salt_pepper_noise[mask]
            elif nt == "sine":
                freq = noise_intensity[nt].get("freq", 1)
                amp = noise_intensity[nt].get("amp", 1)
                phase = noise_intensity[nt].get("phase", 0)
                time_steps = torch.arange(
                    input_tensor.shape[-1], device=input_tensor.device
                ).float()
                noise = amp * torch.sin(2 * math.pi * freq * time_steps + phase)
                input_tensor += noise
            elif nt == "random_walk":
                step_std = noise_intensity[nt].get("step_std", 1)
                steps = (
                    torch.randn(input_tensor.shape[-1], device=input_tensor.device)
                    * step_std
                )
                noise = torch.cumsum(steps, dim=0)
                input_tensor += noise
            else:
                raise ValueError(f"Unknown noise type: {nt}")

        return input_tensor

    def encoding(self, long_term_history, noise=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, C, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            noise (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        if noise:
            if self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape
                # PCA, add noise, and then inverse PCA
                patches = patches.transpose(1, 2)  # B, P, N, d
                if self.low_rank_method == "pca":
                    reduced, U_k, S_k, Vh_k, mean = self.batch_pca(
                        patches.reshape(-1, num_nodes, num_dim), self.low_rank
                    )
                elif self.low_rank_method == "ae":
                    reduced = self.ae(patches, mode="rep")
                else:
                    raise ValueError("Unknown low rank method.")
                reduced_noised = self.add_noise(
                    reduced, self.noise_type, self.noise_intensity
                )
                if self.low_rank_method == "pca":
                    reconstructed_patches = self.batch_inv_pca(
                        reduced_noised, U_k, S_k, Vh_k, mean
                    )
                elif self.low_rank_method == "ae":
                    reconstructed_patches = self.ae(reduced_noised, mode="dec")
                else:
                    raise ValueError("Unknown low rank method.")
                reconstruction_loss = torch.mean((reconstructed_patches - patches) ** 2)

                reconstructed_patches = reconstructed_patches.view(
                    batch_size, num_time, num_nodes, num_dim
                )
                reconstructed_patches = reconstructed_patches.transpose(
                    1, 2
                )  # B, N, P, d
                # positional embedding
                reconstructed_patches, self.pos_mat = self.positional_encoding(
                    reconstructed_patches
                )  # mask
                reconstructed_patches = reconstructed_patches.transpose(
                    1, 2
                )  # B, P, N, d
                encoder_input = reconstructed_patches
                # print(encoder_input.shape)
                hidden_states = self.encoder(encoder_input)
                hidden_states = self.encoder_norm(hidden_states).view(
                    batch_size, -1, num_nodes, self.embed_dim
                )  # B, P, N, d

            if not self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape
                # PCA, add noise, and then inverse PCA
                if self.low_rank_method == "pca":
                    reduced, U_k, S_k, Vh_k, mean = self.batch_pca(
                        patches.view(-1, num_time, num_dim), self.low_rank
                    )
                elif self.low_rank_method == "ae":
                    reduced = self.ae(patches, mode="rep")
                else:
                    raise ValueError("Unknown low rank method.")

                pca_reduced_noised = self.add_noise(
                    reduced, self.noise_type, self.noise_intensity
                )
                if self.low_rank_method == "pca":
                    reconstructed_patches = self.batch_inv_pca(
                        pca_reduced_noised, U_k, S_k, Vh_k, mean
                    )
                elif self.low_rank_method == "ae":
                    reconstructed_patches = self.ae(pca_reduced_noised, mode="dec")
                else:
                    raise ValueError("Unknown low rank method.")
                reconstruction_loss = torch.mean((reconstructed_patches - patches) ** 2)

                reconstructed_patches = reconstructed_patches.view(
                    batch_size, num_nodes, num_time, num_dim
                )  # B, N, P, d
                # positional embedding
                reconstructed_patches, self.pos_mat = self.positional_encoding(
                    reconstructed_patches
                )  ## mask
                encoder_input = reconstructed_patches
                encoder_input = encoder_input  # .transpose(-2, -3) # B, N, P, d
                # print(encoder_input.shape)
                hidden_states = self.encoder(encoder_input)
                hidden_states = self.encoder_norm(hidden_states).view(
                    batch_size, -1, num_time, self.embed_dim
                )  # B, N, P, d

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            # patchify and embed input
            patches = self.patch_embedding(long_term_history)  # B, N, d, P
            patches = patches.transpose(-1, -2)  # B, N, P, d
            # positional embedding
            patches, self.pos_mat = self.positional_encoding(patches)  # B, N, P, d
            # print(self.pos_mat.shape)
            encoder_input = patches  # B, N, P, d
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)  # B, P, N, d
            hidden_states = self.encoder(encoder_input)  # B,  P,N, d/# B, N, P, d
            if self.spatial:
                hidden_states = hidden_states.transpose(-2, -3)  # B, N, P, d
            hidden_states = self.encoder_norm(hidden_states).view(
                batch_size, num_nodes, -1, self.embed_dim
            )  # B, N, P, d
            return hidden_states, None
        # encoding

        return hidden_states, reconstruction_loss

    def decoding(self, hidden_states):

        # encoder 2 decoder layer
        hidden_states = self.enc_2_dec_emb(hidden_states)  # B, N, P, d/# B,P, N,  d
        # B,N*r,P,d
        if self.spatial:  # B, P, N, d
            batch_size, num_time, num_nodes, _ = hidden_states.shape
            hidden_states += self.pos_mat.transpose(1, 2)
            # decoding
            hidden_states = self.decoder(hidden_states)
            hidden_states = self.decoder_norm(hidden_states)

            # prediction (reconstruction)
            reconstruction = self.output_layer(
                hidden_states.view(batch_size, -1, num_nodes, self.embed_dim)
            ).transpose(
                1, 2
            )  # B, N, P, d

        else:
            batch_size, num_nodes, num_time, _ = hidden_states.shape  # B, N, P, d
            hidden_states += self.pos_mat  # B, N, P, d

            # decoding
            hidden_states = self.decoder(hidden_states)  # B, N, P, d
            hidden_states = self.decoder_norm(hidden_states)  # B, N, P, d
            # prediction (reconstruction)
            reconstruction = self.output_layer(
                hidden_states.view(batch_size, num_nodes, -1, self.embed_dim)
            )  # B, N, P, d

        return reconstruction

    def forward(self, history_data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P

        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states, reconstruction_loss = self.encoding(history_data)
            # decoding
            reconstruction = self.decoding(hidden_states)
            reconstruction = reconstruction.reshape(
                -1, history_data.shape[1], history_data.shape[2], history_data.shape[3]
            )
            return reconstruction, history_data, reconstruction_loss
        else:
            hidden_states_full, _ = self.encoding(history_data, noise=False)
            return hidden_states_full


def main():
    import sys
    from torchsummary import summary

    GPU = sys.argv[-1] if len(sys.argv) == 2 else "2"
    device = (
        torch.device("cuda:{}".format(GPU))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = STDNoise(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        noise_type="gaussian",
        noise_intensity=0.1,
        encoder_depth=4,
        decoder_depth=1,
        mode="pre-train",
    ).to(device)
    summary(model, (288 * 7, 307, 1), device=device)


if __name__ == "__main__":
    main()
