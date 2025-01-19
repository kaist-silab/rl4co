from dataclasses import dataclass, fields
from typing import Tuple

import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Tensor | float
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


class AttentionModelDecoder(AutoregressiveDecoder):
    """
    Auto-regressive decoder based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    In this case we additionally have a `pre_decoder_hook` method that allows to precompute the embeddings before
    the decoder is called, which saves a lot of computation.


    Args:
        embed_dim: Embedding dimension
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
        pointer: Module implementing the pointer logic (defaults to PointerAttention)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

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

        if pointer is None:
            # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
            pointer_attn_class = (
                PointerAttention if moe_kwargs is None else PointerAttnMoE
            )
            pointer = pointer_attn_class(
                embed_dim,
                num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )

        self.pointer = pointer

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        return glimpse_k, glimpse_v, logit_k

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Compute logits
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )
