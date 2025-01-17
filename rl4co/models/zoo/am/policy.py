from typing import Callable

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder


class AttentionModelPolicy(AutoregressivePolicy):
    """
    Attention Model Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    This model first encodes the input graph using a Graph Attention Network (GAT) (:class:`AttentionModelEncoder`)
    and then decodes the solution using a pointer network (:class:`AttentionModelDecoder`). Cache is used to store the
    embeddings of the nodes to be used by the decoder to save computation.
    See :class:`rl4co.models.common.constructive.autoregressive.policy.AutoregressivePolicy` for more details on the inference process.

    Args:
        encoder: Encoder module, defaults to :class:`AttentionModelEncoder`
        decoder: Decoder module, defaults to :class:`AttentionModelDecoder`
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        encoder_network: Network to use for the encoder
        init_embedding: Module to use for the initialization of the embeddings
        context_embedding: Module to use for the context embedding
        dynamic_embedding: Module to use for the dynamic embedding
        use_graph_context: Whether to use the graph context
        linear_bias_decoder: Whether to use a bias in the linear layer of the decoder
        sdpa_fn_encoder: Function to use for the scaled dot product attention in the encoder
        sdpa_fn_decoder: Function to use for the scaled dot product attention in the decoder
        sdpa_fn: (deprecated) Function to use for the scaled dot product attention
        mask_inner: Whether to mask the inner product
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        check_nan: Whether to check for nan values during decoding
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        mask_logits: Whether to mask the logits during decoding
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
        moe_kwargs: Keyword arguments for MoE,
            e.g., {"encoder": {"hidden_act": "ReLU", "num_experts": 4, "k": 2, "noisy_gating": True},
                   "decoder": {"light_version": True, ...}}
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "tsp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        **unused_kwargs,
    ):
        if encoder is None:
            encoder = AttentionModelEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
                moe_kwargs=moe_kwargs["encoder"],
            )

        if decoder is None:
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                moe_kwargs=moe_kwargs["decoder"],
            )

        super(AttentionModelPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
