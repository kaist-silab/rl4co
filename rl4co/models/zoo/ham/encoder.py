import torch.nn as nn

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import Normalization, SkipConnection
from rl4co.models.zoo.ham.attention import HeterogenousMHA


class HeterogeneuousMHALayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        feedforward_hidden=512,
        normalization="batch",
    ):
        super(HeterogeneuousMHALayer, self).__init__(
            SkipConnection(HeterogenousMHA(num_heads, embed_dim, embed_dim)),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feedforward_hidden),
                    nn.ReLU(),
                    nn.Linear(feedforward_hidden, embed_dim),
                )
                if feedforward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class GraphHeterogeneousAttentionEncoder(nn.Module):
    def __init__(
        self,
        init_embedding=None,
        num_heads=8,
        embed_dim=128,
        num_encoder_layers=3,
        env_name=None,
        normalization="batch",
        feedforward_hidden=512,
        sdpa_fn=None,
    ):
        super(GraphHeterogeneousAttentionEncoder, self).__init__()

        # substitute env_name with pdp if none
        if env_name is None:
            env_name = "pdp"
        # Map input to embedding space
        if init_embedding is None:
            self.init_embedding = env_init_embedding(env_name, {"embed_dim": embed_dim})
        else:
            self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                HeterogeneuousMHALayer(
                    num_heads,
                    embed_dim,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_encoder_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)  # (batch_size, graph_size, embed_dim)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
