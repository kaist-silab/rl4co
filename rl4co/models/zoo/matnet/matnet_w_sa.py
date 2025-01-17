import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import Normalization, TransformerFFN


def apply_weights_and_combine(dots, v, tanh_clipping=0):
    # scale to avoid numerical underflow
    logits = dots / dots.std()
    if tanh_clipping > 0:
        # tanh clipping to avoid explosions
        logits = torch.tanh(logits) * tanh_clipping
    # shape: (batch, num_heads, row_cnt, col_cnt)
    weights = nn.Softmax(dim=-1)(logits)
    weights = weights.nan_to_num(0)
    # shape: (batch, num_heads, row_cnt, qkv_dim)
    out = torch.matmul(weights, v)
    # shape: (batch, row_cnt, num_heads, qkv_dim)
    out = rearrange(out, "b h s d -> b s (h d)")
    return out


class MixedScoreFF(nn.Module):
    def __init__(self, num_heads, ms_hidden_dim: int = 32, bias: bool = False) -> None:
        super().__init__()

        self.lin1 = nn.Linear(2 * num_heads, num_heads * ms_hidden_dim, bias=bias)
        self.lin2 = nn.Linear(num_heads * ms_hidden_dim, num_heads, bias=bias)

    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=-1)
        two_scores = rearrange(two_scores, "b h r c s -> b r c (h s)")
        # shape: (batch, row_cnt, col_cnt, 2 * num_heads)
        ms1 = self.lin1(two_scores)
        ms1_activated = F.relu(ms1)
        # shape: (batch, row_cnt, col_cnt, num_heads)
        ms2 = self.lin2(ms1_activated)
        # shape: (batch, row_cnt, head_num, col_cnt)
        mixed_scores = rearrange(ms2, "b r c h -> b h r c")

        return mixed_scores


class EfficientMixedScoreMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()

        qkv_dim = embed_dim // num_heads

        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)

        self.Wqv1 = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.Wkv2 = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)

        # self.init_parameters()
        self.mixed_scores_layer = MixedScoreFF(num_heads, qkv_dim, bias)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x1, x2, attn_mask=None, cost_mat=None):
        batch_size = x1.size(0)
        row_cnt = x1.size(-2)
        col_cnt = x2.size(-2)

        # Project query, key, value
        q, v1 = rearrange(
            self.Wqv1(x1), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # Project query, key, value
        k, v2 = rearrange(
            self.Wqv1(x2), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # shape: (batch, num_heads, row_cnt, col_cnt)
        dot = self.norm_factor * torch.matmul(q, k.transpose(-2, -1))

        if cost_mat is not None:
            # shape: (batch, num_heads, row_cnt, col_cnt)
            cost_mat_score = cost_mat[:, None, :, :].expand_as(dot)
            dot = self.mixed_scores_layer(dot, cost_mat_score)

        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(dot)
            dot.masked_fill_(~attn_mask, float("-inf"))

        h1 = self.out_proj1(apply_weights_and_combine(dot, v2))
        h2 = self.out_proj2(apply_weights_and_combine(dot.transpose(-2, -1), v1))

        return h1, h2


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: str = "batch",
        bias: bool = False,
    ):
        super().__init__()

        self.op_attn = MultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.ma_attn = MultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.cross_attn = EfficientMixedScoreMultiHeadAttention(
            embed_dim, num_heads, bias=bias
        )

        self.op_ffn = TransformerFFN(embed_dim, feedforward_hidden, normalization)
        self.ma_ffn = TransformerFFN(embed_dim, feedforward_hidden, normalization)

        self.op_norm = Normalization(embed_dim, normalization)
        self.ma_norm = Normalization(embed_dim, normalization)

    def forward(
        self, op_in, ma_in, cost_mat, op_mask=None, ma_mask=None, cross_mask=None
    ):
        op_cross_out, ma_cross_out = self.cross_attn(
            op_in, ma_in, attn_mask=cross_mask, cost_mat=cost_mat
        )
        op_cross_out = self.op_norm(op_cross_out + op_in)
        ma_cross_out = self.ma_norm(ma_cross_out + ma_in)

        # (bs, num_jobs, ops_per_job, d)
        op_self_out = self.op_attn(op_cross_out, attn_mask=op_mask)
        # (bs, num_ma, d)
        ma_self_out = self.ma_attn(ma_cross_out, attn_mask=ma_mask)

        op_out = self.op_ffn(op_cross_out, op_self_out)
        ma_out = self.ma_ffn(ma_cross_out, ma_self_out)

        return op_out, ma_out


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 5,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = embed_dim

        if init_embedding is None:
            init_embedding = env_init_embedding(
                "matnet", {"embed_dim": embed_dim, **init_embedding_kwargs}
            )
        self.init_embedding = init_embedding
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td, attn_mask: torch.Tensor = None):
        # [BS, num_machines, emb], [BS, num_operations, emb]
        ops_embed, ma_embed, edge_feat = self.init_embedding(td)
        try:
            # mask padded ops; shape=(bs, ops)
            ops_attn_mask = ~td["pad_mask"]
        except KeyError:
            ops_attn_mask = None
        # padded ops should also be masked in cross attention; shape=(bs, ops, ma)
        # cross_mask = ops_attn_mask.unsqueeze(-1).expand(-1, -1, ma_embed.size(1))
        for layer in self.layers:
            ops_embed, ma_embed = layer(
                ops_embed,
                ma_embed,
                cost_mat=edge_feat,
                op_mask=ops_attn_mask,  # mask padded operations in attention
                ma_mask=None,  # no padding for machines
                cross_mask=None,
            )
        embedding = (ops_embed, ma_embed)
        return embedding, None
