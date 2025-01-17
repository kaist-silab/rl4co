from typing import Callable, Optional

import torch.nn as nn

from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.ham.encoder import GraphHeterogeneousAttentionEncoder


class HeterogeneousAttentionModelPolicy(AttentionModelPolicy):
    """Heterogeneous Attention Model Policy based on https://ieeexplore.ieee.org/document/9352489.
    We re-declare the most important arguments here for convenience as in the paper.
    See :class:`rl4co.models.zoo.am.AttentionModelPolicy` for more details.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        env_name: Name of the environment used to initialize embeddings
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        embed_dim: Dimension of the embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads for the attention in encoder
        normalization: Normalization to use for the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        sdpa_fn: Function to use for the scaled dot product attention
        **kwargs: keyword arguments passed to the :class:`rl4co.models.zoo.am.AttentionModelPolicy`
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        env_name: str = "pdp",
        init_embedding: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        sdpa_fn: Optional[Callable] = None,
        **kwargs,
    ):
        if encoder is None:
            encoder = GraphHeterogeneousAttentionEncoder(
                init_embedding=init_embedding,
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                sdpa_fn=sdpa_fn,
            )
        else:
            encoder = encoder

        super(HeterogeneousAttentionModelPolicy, self).__init__(
            env_name=env_name,
            encoder=encoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
