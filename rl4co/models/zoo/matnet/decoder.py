from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.nn.env_embeddings.context import FFSPContext
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.utils.decoding import decode_logprobs, process_logits
from rl4co.utils.ops import gather_by_index


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor | TensorDict
    graph_context: Tensor | float
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class MatNetDecoder(AttentionModelDecoder):
    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], *args, **kwargs):
        row_emb, col_emb = embeddings
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


class MatNetFFSPDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        linear_bias: bool = False,
        out_bias_pointer_attn: bool = True,
        use_graph_context: bool = False,
        **kwargs,
    ):
        context_embedding = FFSPContext(embed_dim)

        super().__init__(
            env_name="ffsp",
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_embedding=context_embedding,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias,
            use_graph_context=use_graph_context,
            **kwargs,
        )

        self.no_job_emb = nn.Parameter(torch.rand(1, 1, embed_dim), requires_grad=True)

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], **kwargs):
        job_emb, ma_emb = embeddings

        bs, _, emb_dim = job_emb.shape

        job_emb_plus_one = torch.cat(
            (job_emb, self.no_job_emb.expand((bs, 1, emb_dim))), dim=1
        )

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            job_emb_plus_one
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(job_emb_plus_one.mean(1))
        else:
            graph_context = 0

        embeddings = TensorDict(
            {"job_embeddings": job_emb_plus_one, "machine_embeddings": ma_emb},
            batch_size=bs,
        )
        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )


class MultiStageFFSPDecoder(MatNetFFSPDecoder):
    """Decoder class for the solving the FFSP using a seperate MatNet decoder for each stage
    as originally implemented by Kwon et al. (2021)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_graph_context: bool = True,
        tanh_clipping: float = 10,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
            **kwargs,
        )
        self.cached_embs: PrecomputedCache = None
        self.tanh_clipping = tanh_clipping

    def _precompute_cache(self, embeddings: Tuple[Tensor], **kwargs):
        self.cached_embs = super()._precompute_cache(embeddings, **kwargs)

    def forward(
        self,
        td: TensorDict,
        decode_type="sampling",
        num_starts: int = 1,
        **decoding_kwargs,
    ) -> Tuple[Tensor, Tensor, TensorDict]:

        logits, mask = super().forward(td, self.cached_embs, num_starts)
        logprobs = process_logits(
            logits,
            mask,
            tanh_clipping=self.tanh_clipping,
            **decoding_kwargs,
        )
        job_selected = decode_logprobs(logprobs, mask, decode_type)
        job_prob = gather_by_index(logprobs, job_selected, dim=1)

        return job_selected, job_prob
