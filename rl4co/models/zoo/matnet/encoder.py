from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadCrossAttention
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import TransformerFFN


class MixedScoresSDPA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_scores: int = 1,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_scores = num_scores
        mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, self.num_scores + 1, mixer_hidden_dim)
        )
        mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, mixer_hidden_dim)
        )
        self.mix_W1 = nn.Parameter(mix_W1)
        self.mix_b1 = nn.Parameter(mix_b1)

        mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, mixer_hidden_dim, 1)
        )
        mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, 1)
        )
        self.mix_W2 = nn.Parameter(mix_W2)
        self.mix_b2 = nn.Parameter(mix_b2)

    def forward(self, q, k, v, attn_mask=None, dmat=None, dropout_p=0.0):
        """Scaled Dot-Product Attention with MatNet Scores Mixer"""
        assert dmat is not None
        b, m, n = dmat.shape[:3]
        dmat = dmat.reshape(b, m, n, self.num_scores)

        # Calculate scaled dot product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        # [b, h, m, n, num_scores+1]
        mix_attn_scores = torch.cat(
            [
                attn_scores.unsqueeze(-1),
                dmat[:, None, ...].expand(b, self.num_heads, m, n, self.num_scores),
            ],
            dim=-1,
        )
        # [b, h, m, n]
        attn_scores = (
            (
                torch.matmul(
                    F.relu(
                        torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
                        + self.mix_b1[None, None, :, None, :]
                    ),
                    self.mix_W2,
                )
                + self.mix_b2[None, None, :, None, :]
            )
            .transpose(1, 2)
            .squeeze(-1)
        )

        # Apply the provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask[~attn_mask.any(-1)] = True
                attn_scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)


class MatNetCrossMHA(MultiHeadCrossAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        attn_fn = MixedScoresSDPA(
            num_heads=num_heads,
            mixer_hidden_dim=mixer_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init,
        )

        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, bias=bias, sdpa_fn=attn_fn
        )


class MatNetMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.row_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)
        self.col_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """
        updated_row_emb = self.row_encoding_block(
            row_emb, col_emb, dmat=dmat, cross_attn_mask=attn_mask
        )
        attn_mask_t = attn_mask.transpose(-2, -1) if attn_mask is not None else None
        updated_col_emb = self.col_encoding_block(
            col_emb,
            row_emb,
            dmat=dmat.transpose(-2, -1),
            cross_attn_mask=attn_mask_t,
        )
        return updated_row_emb, updated_col_emb


class MatNetLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
    ):
        super().__init__()
        self.MHA = MatNetMHA(embed_dim, num_heads, bias)
        self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
        self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        row_emb_out, col_emb_out = self.MHA(row_emb, col_emb, dmat, attn_mask)
        row_emb_out = self.F_a(row_emb_out, row_emb)
        col_emb_out = self.F_b(col_emb_out, col_emb)
        return row_emb_out, col_emb_out


class MatNetEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        bias: bool = False,
        mask_non_neighbors: bool = False,
    ):
        super().__init__()

        if init_embedding is None:
            init_embedding = env_init_embedding(
                "matnet", {"embed_dim": embed_dim, **init_embedding_kwargs}
            )

        self.init_embedding = init_embedding
        self.mask_non_neighbors = mask_non_neighbors
        self.layers = nn.ModuleList(
            [
                MatNetLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=bias,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td, attn_mask: torch.Tensor = None):
        row_emb, col_emb, dmat = self.init_embedding(td)

        if self.mask_non_neighbors and attn_mask is None:
            # attn_mask (keep 1s discard 0s) to only attend on neighborhood
            attn_mask = dmat.ne(0)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, dmat, attn_mask)

        embedding = (row_emb, col_emb)
        init_embedding = None
        return embedding, init_embedding  # match output signature for the AR policy class
