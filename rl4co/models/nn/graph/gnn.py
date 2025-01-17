import torch
import torch.nn as nn

try:
    import torch_geometric.nn as gnn
except ImportError:
    gnn = None

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GNNLayer(nn.Module):
    """Graph Neural Network Layer for processing graph structures.

    Args:
        units: The number of units in each linear transformation layer.
        act_fn: The name of the activation function to use after each linear layer. Defaults to 'silu'.
        agg_fn: The name of the global aggregation function to use for pooling features across the graph. Defaults to 'mean'.
    """

    def __init__(self, units: int, act_fn: str = "silu", agg_fn: str = "mean"):
        assert gnn is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(GNNLayer, self).__init__()
        self.units = units
        self.act_fn = getattr(nn.functional, act_fn)
        self.agg_fn = getattr(gnn, f"global_{agg_fn}_pool")

        # Vertex updates
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)

        # Edge updates
        self.e_lin = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, edge_index, edge_attr):
        x0 = x
        w0 = w = edge_attr

        # Vertex updates
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        x = x0 + self.act_fn(
            self.v_bn(
                x1 + self.agg_fn(torch.sigmoid(w0) * x2[edge_index[1]], edge_index[0])
            )
        )

        # Edge updates
        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


class GNNEncoder(nn.Module):
    """Anisotropic Graph Neural Network encoder with edge-gating mechanism as in Joshi et al. (2022)

    Args:
        num_layers: The number of GNN layers to stack in the network.
        embed_dim: The dimensionality of the embeddings for each node in the graph.
        act_fn: The activation function to use in each GNNLayer, see https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions for available options. Defaults to 'silu'.
        agg_fn: The aggregation function to use in each GNNLayer for pooling features. Options: 'add', 'mean', 'max'. Defaults to 'mean'.
    """

    def __init__(self, num_layers: int, embed_dim: int, act_fn="silu", agg_fn="mean"):
        super(GNNEncoder, self).__init__()
        self.act_fn = getattr(nn.functional, act_fn)
        self.agg_fn = agg_fn

        # Stack of GNN layers
        self.layers = nn.ModuleList(
            [GNNLayer(embed_dim, act_fn, agg_fn) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, w):
        """Sequentially passes the input graph data through the stacked GNN layers,
        applying specified transformations and aggregations to learn graph representations.

        Args:
            x: The node features of the graph with shape [num_nodes, embed_dim].
            edge_index: The edge indices of the graph with shape [2, num_edges].
            w: The edge attributes or weights with shape [num_edges, embed_dim].
        """
        x = self.act_fn(x)
        w = self.act_fn(w)
        for layer in self.layers:
            x, w = layer(x, edge_index, w)
        return x, w
