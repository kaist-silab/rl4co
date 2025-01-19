from typing import Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MessagePassingLayer(MessagePassing):
    def __init__(
        self,
        node_indim,
        node_outdim,
        edge_indim,
        edge_outdim,
        aggregation="add",
        residual=False,
        **mlp_params,
    ):
        super(MessagePassingLayer, self).__init__(aggr=aggregation)
        # Init message passing models
        self.edge_model = MLP(
            input_dim=edge_indim + 2 * node_indim, output_dim=edge_outdim, **mlp_params
        )
        self.node_model = MLP(
            input_dim=edge_outdim + node_indim, output_dim=node_outdim, **mlp_params
        )
        self.residual = residual

    def forward(self, node_feature, edge_feature, edge_index, mask=None):
        # Message passing
        update_edge_feature = self.edge_update(node_feature, edge_feature, edge_index)
        update_node_feature = self.propagate(
            edge_index, x=node_feature, edge_features=update_edge_feature
        )

        # Update with residual connection
        if self.residual:
            update_node_feature = update_node_feature + node_feature

        return update_node_feature, update_edge_feature

    def edge_update(self, nf, ef, edge_index):
        row, col = edge_index
        x_i, x_j = nf[row], nf[col]
        uef = self.edge_model(torch.cat([x_i, x_j, ef], dim=-1))
        return uef

    def message(self, edge_features: torch.tensor):
        return edge_features

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf


class MessagePassingEncoder(nn.Module):
    def __init__(
        self,
        env_name: str,
        embed_dim: int,
        num_nodes: int,
        num_layers: int,
        init_embedding: nn.Module = None,
        aggregation: str = "add",
        self_loop: bool = False,
        residual: bool = True,
    ):
        """
        Note:
            - Support fully connected graph for now.
        """
        super(MessagePassingEncoder, self).__init__()

        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        # Generate edge index for a fully connected graph
        adj_matrix = torch.ones(num_nodes, num_nodes)
        if self_loop:
            adj_matrix.fill_diagonal_(0)  # No self-loops
        self.edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))

        # Init message passing models
        self.mpnn_layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_indim=embed_dim,
                    node_outdim=embed_dim,
                    edge_indim=1,
                    edge_outdim=1,
                    aggregation=aggregation,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        )

        # Record parameters
        self.self_loop = self_loop

    # def forward(self, x, mask=None):
    def forward(
        self, td: TensorDict, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)
        num_node = init_h.size(-2)

        # Check to update the edge index with different number of node
        if num_node != self.edge_index.max().item() + 1:
            adj_matrix = torch.ones(num_node, num_node)
            if self.self_loop:
                adj_matrix.fill_diagonal_(0)
            edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))
            edge_index = edge_index.to(init_h.device)
        else:
            edge_index = self.edge_index.to(init_h.device)

        # Generate edge features: distance
        edge_feature = torch.norm(
            init_h[..., edge_index[0], :] - init_h[..., edge_index[1], :],
            dim=-1,
            keepdim=True,
        )

        # Create the batched graph
        data_list = [
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            for x, edge_attr in zip(init_h, edge_feature)
        ]
        data_batch = Batch.from_data_list(data_list)
        update_node_feature = data_batch.x
        update_edge_feature = data_batch.edge_attr
        edge_index = data_batch.edge_index

        # Message passing
        for layer in self.mpnn_layers:
            update_node_feature, update_edge_feature = layer(
                update_node_feature, update_edge_feature, edge_index
            )

        # De-batch the graph
        input_size = init_h.size()
        update_node_feature = update_node_feature.view(*input_size)

        return update_node_feature, init_h

    def edge_update(self, nf, ef, edge_index):
        row, col = edge_index
        x_i, x_j = nf[row], nf[col]
        uef = self.edge_model(torch.cat([x_i, x_j, ef], dim=-1))
        return uef

    def message(self, edge_features: torch.tensor):
        return edge_features

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf
