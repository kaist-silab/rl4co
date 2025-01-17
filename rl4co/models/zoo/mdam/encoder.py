from typing import Callable, Optional

import torch
import torch.nn as nn

from rl4co.models.nn.graph.attnnet import (
    MultiHeadAttentionLayer,
    Normalization,
    SkipConnection,
)
from rl4co.models.zoo.mdam.mha import MultiHeadAttentionMDAM


class MDAMGraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_layers,
        node_dim=None,
        normalization="batch",
        feedforward_hidden=512,
        sdpa_fn: Optional[Callable] = None,
    ):
        super(MDAMGraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    embed_dim,
                    num_heads,
                    feedforward_hidden,
                    normalization,
                    sdpa_fn=sdpa_fn,
                )
                for _ in range(num_layers - 1)  # because last layer is different
            )
        )
        self.attention_layer = MultiHeadAttentionMDAM(
            embed_dim, num_heads, sdpa_fn=sdpa_fn, last_one=True
        )
        self.BN1 = Normalization(embed_dim, normalization)
        self.projection = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feedforward_hidden),
                nn.ReLU(),
                nn.Linear(feedforward_hidden, embed_dim),
            )
            if feedforward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.BN2 = Normalization(embed_dim, normalization)

    def forward(self, x, mask=None, return_transform_loss=False):
        """
        Returns:
            - h [batch_size, graph_size, embed_dim]
            - attn [num_head, batch_size, graph_size, graph_size]
            - V [num_head, batch_size, graph_size, key_dim]
            - h_old [batch_size, graph_size, embed_dim]
        """
        assert mask is None, "TODO mask not yet supported!"

        h_embeded = x
        h_old = self.layers(h_embeded)
        h_new, attn, V = self.attention_layer(h_old)
        h = h_new + h_old
        h = self.BN1(h)
        h = self.projection(h)
        h = self.BN2(h)

        return (h, h.mean(dim=1), attn, V, h_old)

    def change(self, attn, V, h_old, mask):
        num_heads, batch_size, graph_size, feat_size = V.size()
        attn = (
            mask.float()
            .view(1, batch_size, 1, graph_size)
            .repeat(num_heads, 1, graph_size, 1)
            * attn
        )
        attn = attn / (
            torch.sum(attn, dim=-1).view(num_heads, batch_size, graph_size, 1) + 1e-9
        )
        heads = torch.matmul(attn, V)

        h_new = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.attention_layer.num_heads * self.attention_layer.val_dim),
            self.attention_layer.W_out.view(-1, self.attention_layer.embed_dim),
        ).view(batch_size, graph_size, self.attention_layer.embed_dim)
        h = h_new + h_old
        h = self.BN1(h)
        h = self.projection(h)
        h = self.BN2(h)

        return (h, h.mean(dim=1))
