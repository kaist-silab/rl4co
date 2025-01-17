from typing import Optional

import torch.nn as nn

from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveDecoder,
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.utils.pylogger import get_pylogger

from .encoder import NARGNNEncoder

log = get_pylogger(__name__)


class NARGNNPolicy(NonAutoregressivePolicy):
    """
    Base Non-autoregressive policy for NCO construction methods.
    This creates a heatmap of NxN for N nodes (i.e., heuristic) that models the probability to go from one node to another for all nodes.

    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution to the NCO problem

    Warning:
        The effectiveness of the non-autoregressive approach can vary significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and decoding strategy to achieve competitive results.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        decoder: Decoder module. Note that this moule defaults to the non-autoregressive decoder
        embed_dim: Dimension of the embeddings
        env_name: Name of the environment used to initialize embeddings
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        graph_network: Model to use for the graph network. If None, use the default embedding for the environment
        heatmap_generator: Model to use for the heatmap generator. If None, use the default embedding for the environment
        num_layers_heatmap_generator: Number of layers in the heatmap generator
        num_layers_graph_encoder: Number of layers in the graph encoder
        act_fn: Activation function to use in the encoder
        agg_fn: Aggregation function to use in the encoder
        linear_bias: Whether to use bias in the encoder
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **constructive_policy_kw: Unused keyword arguments
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn="silu",
        agg_fn="mean",
        linear_bias: bool = True,
        train_decode_type: str = "multistart_sampling",
        val_decode_type: str = "multistart_greedy",
        test_decode_type: str = "multistart_greedy",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            encoder = NARGNNEncoder(
                embed_dim=embed_dim,
                env_name=env_name,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
                graph_network=graph_network,
                heatmap_generator=heatmap_generator,
                num_layers_heatmap_generator=num_layers_heatmap_generator,
                num_layers_graph_encoder=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
                linear_bias=linear_bias,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = NonAutoregressiveDecoder()
        else:
            # check if the decoder has trainable parameters
            if any(p.requires_grad for p in decoder.parameters()):
                log.error(
                    "The decoder contains trainable parameters. This should not happen in a non-autoregressive policy."
                )

        # Pass to constructive policy
        super(NARGNNPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
