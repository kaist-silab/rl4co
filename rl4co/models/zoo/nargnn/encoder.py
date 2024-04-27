from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive import NonAutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_edge_embedding, env_init_embedding
from rl4co.models.nn.graph.gnn import GNNEncoder

try:
    from torch_geometric.data import Batch
except ImportError:
    # `Batch` is referred to only as type notations in this file
    Batch = None


class EdgeHeatmapGenerator(nn.Module):
    """MLP for converting edge embeddings to heatmaps

    Args:
        embed_dim: Dimension of the embeddings
        num_layers: The number of linear layers in the network.
        act_fn: Activation function. Defaults to "silu".
        linear_bias: Use bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        act_fn: Union[str, Callable] = "silu",
        linear_bias: bool = True,
        undirected_graph: bool = True,
    ) -> None:
        super(EdgeHeatmapGenerator, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_dim, embed_dim, bias=linear_bias)
                for _ in range(num_layers - 1)
            ]
        )
        self.output = nn.Linear(embed_dim, 1, bias=linear_bias)

        self.act = getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn

        self.undirected_graph = undirected_graph

    def forward(self, graph: Batch) -> Tensor:  # type: ignore
        # do not reuse the input value
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr)) * 10  # type: ignore

        heatmaps_logits = self._make_heatmaps(graph)
        return heatmaps_logits

    def _make_heatmaps(self, batch_graph: Batch) -> Tensor:  # type: ignore
        graphs = batch_graph.to_data_list()
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmaps_logits = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmaps_logits[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        if self.undirected_graph:
            heatmaps_logits = (heatmaps_logits + heatmaps_logits.transpose(1, 2)) * 0.5

        return heatmaps_logits


class NARGNNEncoder(NonAutoregressiveEncoder):
    """Anisotropic Graph Neural Network encoder with edge-gating mechanism as in Joshi et al. (2022), and used in DeepACO (Ye et al., 2023)
    This creates a heatmap # TODO

        This model utilizes a multi-layer perceptron (MLP) approach to predict edge attributes directly from the input graph features,
    which are then transformed into a heatmap representation to facilitate the decoding of the solution. The decoding process
    is managed by a specified strategy which could vary from simple greedy selection to more complex sampling methods.

    # TODO
    Tip:
        This decoder's performance heavily relies on the ability of the MLP to capture the dependencies between different
        parts of the solution without the iterative refinement provided by autoregressive models. It is particularly useful
        in scenarios where the solution space can be effectively explored in a parallelized manner or when the solution components
        are largely independent.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embed_dim: Dimension of the node embeddings
        num_layers: Number of layers in the encoder
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        act_fn: The activation function to use in each GNNLayer, see https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions for available options. Defaults to 'silu'.
        agg_fn: The aggregation function to use in each GNNLayer for pooling features. Options: 'add', 'mean', 'max'. Defaults to 'mean'.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        # TODO: pass network
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn="silu",
        agg_fn="mean",
        linear_bias: bool = True,
    ):
        super(NonAutoregressiveEncoder, self).__init__()
        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        self.edge_embedding = (
            env_edge_embedding(self.env_name, {"embed_dim": embed_dim})
            if edge_embedding is None
            else edge_embedding
        )

        self.graph_network = (
            GNNEncoder(
                embed_dim=embed_dim,
                num_layers=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
            )
            if graph_network is None
            else graph_network
        )

        self.heatmap_generator = (
            EdgeHeatmapGenerator(
                embed_dim=embed_dim,
                num_layers=num_layers_heatmap_generator,
                linear_bias=linear_bias,
            )
            if heatmap_generator is None
            else heatmap_generator
        )

    def forward(self, td: TensorDict):
        """Forward pass of the encoder.
        Transform the input TensorDict into the latent representation.
        """
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        # Process embedding into graph
        # TODO: standardize?
        graph.x, graph.edge_attr = self.graph_network(
            graph.x, graph.edge_index, graph.edge_attr
        )

        # Generate heatmaps
        heatmaps_logits = self.heatmap_generator(graph)

        # Return latent representation (i.e. heatmap logits) and initial embeddings
        return heatmaps_logits, node_embed