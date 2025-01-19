from typing import Callable, Optional

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.common.constructive.nonautoregressive import NonAutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_edge_embedding, env_init_embedding
from rl4co.models.nn.graph.gnn import GNNEncoder

try:
    from torch_geometric.data import Batch
except ImportError:
    # `Batch` is referred to only as type notations in this file
    Batch = None


class EdgeHeatmapGenerator(nn.Module):
    """MLP for converting edge embeddings to heatmaps.

    Args:
        embed_dim: Dimension of the embeddings
        num_layers: The number of linear layers in the network.
        act_fn: Activation function. Defaults to "silu".
        linear_bias: Use bias in linear layers. Defaults to True.
        undirected_graph: Whether the graph is undirected. Defaults to True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        act_fn: str | Callable = "silu",
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
        graph.edge_attr = torch.sigmoid(self.output(edge_attr))  # type: ignore

        heatmap_logits = self._make_heatmap_logits(graph)
        return heatmap_logits

    def _make_heatmap_logits(self, batch_graph: Batch) -> Tensor:  # type: ignore
        graphs = batch_graph.to_data_list()
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmap = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmap[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        # This is commented out, because it undo the some of the sparsification.
        # if self.undirected_graph:
        #     heatmap = (heatmap + heatmap.transpose(1, 2)) * 0.5

        # Avoid log(0) by adding a small value
        if heatmap.dtype == torch.float32 or heatmap.dtype == torch.bfloat16:
            small_value = 1e-12
        elif heatmap.dtype == torch.float16:
            small_value = 3e-8  # the smallest positive number such that log(small_value) is not -inf
        else:
            raise ValueError(f"Unsupported dtype: {heatmap.dtype}")

        heatmap += small_value
        heatmap_logits = torch.log(heatmap)

        return heatmap_logits


class NARGNNEncoder(NonAutoregressiveEncoder):
    """Anisotropic Graph Neural Network encoder with edge-gating mechanism as in Joshi et al. (2022), and used in DeepACO (Ye et al., 2023).
    This creates a heatmap of NxN for N nodes (i.e., heuristic) that models the probability to go from one node to another for all nodes.
    This model utilizes a multi-layer perceptron (MLP) approach to predict edge attributes directly from the input graph features,
    which are then transformed into a heatmap representation to facilitate the decoding of the solution. The decoding process
    is managed by a specified strategy which could vary from simple greedy selection to more complex sampling methods.

    Tip:
        This decoder's performance heavily relies on the ability of the MLP to capture the dependencies between different
        parts of the solution without the iterative refinement provided by autoregressive models. It is particularly useful
        in scenarios where the solution space can be effectively explored in a parallelized manner or when the solution components
        are largely independent.

    Args:
        embed_dim: Dimension of the node embeddings
        env_name: Name of the environment used to initialize embeddings
        num_layers: Number of layers in the encoder
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        graph_network: Model to use for the graph network. If None, use the default network for the environment
        heatmap_generator: Model to use for the heatmap generator. If None, use the default network for the environment
        num_layers_heatmap_generator: Number of layers in the heatmap generator
        num_layers_graph_encoder: Number of layers in the graph encoder
        act_fn: The activation function to use in each GNNLayer, see https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions for available options. Defaults to 'silu'.
        agg_fn: The aggregation function to use in each GNNLayer for pooling features. Options: 'add', 'mean', 'max'. Defaults to 'mean'.
        linear_bias: Use bias in linear layers. Defaults to True.
        k_sparse: Number of edges to keep for each node. Defaults to None.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
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
        k_sparse: Optional[int] = None,
    ):
        super(NonAutoregressiveEncoder, self).__init__()
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        self.edge_embedding = (
            env_edge_embedding(
                self.env_name, {"embed_dim": embed_dim, "k_sparse": k_sparse}
            )
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

        # Generate heatmap logits
        heatmap_logits = self.heatmap_generator(graph)

        # Return latent representation (i.e. heatmap logits) and initial embeddings
        return heatmap_logits, node_embed


class NARGNNNodeEncoder(NARGNNEncoder):
    """In this case, we just use the node embeddings from the graph
    without transforming them into a heatmap.
    """

    def forward(self, td: TensorDict):
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        # Process embedding into graph
        # TODO: standardize?
        graph.x, graph.edge_attr = self.graph_network(
            graph.x, graph.edge_index, graph.edge_attr
        )

        proc_embeds = graph.x
        batch_size = node_embed.shape[0]
        # reshape proc_embeds from [bs*n, h] to [bs, n, h]
        proc_embeds = proc_embeds.reshape(batch_size, -1, proc_embeds.shape[1])
        return proc_embeds, node_embed
