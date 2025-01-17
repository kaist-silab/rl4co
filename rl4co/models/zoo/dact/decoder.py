import math

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.common.improvement.base import ImprovementDecoder
from rl4co.models.nn.attention import MultiHeadCompat
from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DACTDecoder(ImprovementDecoder):
    """
    DACT decoder based on Ma et al. (2021)
    Given the environment state and the dual sets of embeddings (PFE, NFE embeddings), compute the logits for
    selecting two nodes for the 2-opt local search from the current solution


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.hidden_dim = embed_dim

        # for MHC sublayer (NFE aspect)
        self.compater_node = MultiHeadCompat(
            num_heads, embed_dim, embed_dim, embed_dim, embed_dim
        )

        # for MHC sublayer (PFE aspect)
        self.compater_pos = MultiHeadCompat(
            num_heads, embed_dim, embed_dim, embed_dim, embed_dim
        )

        self.norm_factor = 1 / math.sqrt(1 * self.hidden_dim)

        # for Max-Pooling sublayer
        self.project_graph_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_graph_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # for feed-forward aggregation (FFA)sublayer
        self.value_head = MLP(
            input_dim=2 * self.n_heads,
            output_dim=1,
            num_neurons=[32, 32],
            dropout_probs=[0.05, 0.00],
        )

    def forward(self, td: TensorDict, final_h: Tensor, final_p: Tensor) -> Tensor:
        """Compute the logits of the removing a node pair from the current solution

        Args:
            td: TensorDict with the current environment state
            final_h: final NFE embeddings
            final_p: final pfe embeddings
        """

        batch_size, graph_size, dim = final_h.size()

        # Max-Pooling sublayer
        h_node_refined = self.project_node_node(final_h) + self.project_graph_node(
            final_h.max(1)[0]
        )[:, None, :].expand(batch_size, graph_size, dim)
        h_pos_refined = self.project_node_pos(final_p) + self.project_graph_pos(
            final_p.max(1)[0]
        )[:, None, :].expand(batch_size, graph_size, dim)

        # MHC sublayer
        compatibility = torch.zeros(
            (batch_size, graph_size, graph_size, self.n_heads * 2),
            device=h_node_refined.device,
        )
        compatibility[:, :, :, : self.n_heads] = self.compater_pos(h_pos_refined).permute(
            1, 2, 3, 0
        )
        compatibility[:, :, :, self.n_heads :] = self.compater_node(
            h_node_refined
        ).permute(1, 2, 3, 0)

        # FFA sublater
        return self.value_head(self.norm_factor * compatibility).squeeze(-1)


class CriticDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.project_graph = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.project_node = nn.Linear(self.input_dim, self.input_dim, bias=False)

        self.MLP = MLP(
            input_dim=input_dim,
            output_dim=1,
            num_neurons=[input_dim, input_dim // 2],
            dropout_probs=[0.05, 0.0],
        )

    def forward(self, x: torch.Tensor, hidden=None) -> torch.Tensor:
        # h_wave: (batch_size, graph_size+1, input_size)
        mean_pooling = x.mean(1)  # mean Pooling (batch_size, input_size)
        graph_feature: torch.Tensor = self.project_graph(mean_pooling)[
            :, None, :
        ]  # (batch_size, 1, input_dim/2)
        node_feature: torch.Tensor = self.project_node(
            x
        )  # (batch_size, graph_size+1, input_dim/2)

        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(
            node_feature
        )  # (batch_size, graph_size+1, input_dim/2)

        value = self.MLP(fusion.mean(1))

        return value
