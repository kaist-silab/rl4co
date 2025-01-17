from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import PolyNetAttention
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Tensor | float
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class PolyNetDecoder(AttentionModelDecoder):
    """
    PolyNet decoder for constructing diverse solutions for combinatorial optimization problems.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    We additionally include support for multi-starts as it is more efficient to do so in the decoder as we can
    natively perform the attention computation.

    Args:
        k: Number of strategies to learn ("K" in the PolyNet paper)
        encoder_type: Type of encoder that should be used. "AM" or "MatNet" are supported
        embed_dim: Embedding dimension
        poly_layer_dim: Dimension of the PolyNet layers
        num_heads: Number of attention heads
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        dynamic_embedding: Dynamic embedding module
        mask_inner: Whether to mask the inner loop
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        linear_bias: Whether to use a bias in the linear layer
        use_graph_context: Whether to use the graph context
        check_nan: Whether to check for nan values during decoding
        sdpa_fn: scaled_dot_product_attention function
    """

    def __init__(
        self,
        k: int,
        encoder_type: str,
        embed_dim: int = 128,
        poly_layer_dim: int = 256,
        num_heads: int = 8,
        env_name: str | RL4COEnvBase = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        **unused_kwargs,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_type = encoder_type

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
        self.pointer = PolyNetAttention(
            k,
            embed_dim,
            poly_layer_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def _precompute_cache_matnet(
        self, embeddings: Tuple[Tensor, Tensor], *args, **kwargs
    ):
        col_emb, row_emb = embeddings
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(col_emb.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=row_emb,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], *args, **kwargs):
        if self.encoder_type == "AM":
            return super()._precompute_cache(embeddings, *args, **kwargs)
        elif self.encoder_type == "MatNet":
            return self._precompute_cache_matnet(embeddings, *args, **kwargs)
