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


class NodePairRemovalDecoder(ImprovementDecoder):
    """
    N2S Node-Pair Removal decoder based on Ma et al. (2022)
    Given the environment state and the node embeddings (positional embeddings are discarded), compute the logits for
    selecting a pair of pickup and delivery nodes for node pair removal from the current solution


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = embed_dim
        self.n_heads = num_heads
        self.hidden_dim = embed_dim

        assert embed_dim % num_heads == 0

        self.W_Q = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.hidden_dim)
        )
        self.W_K = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.hidden_dim)
        )

        self.agg = MLP(input_dim=2 * self.n_heads + 4, output_dim=1, num_neurons=[32, 32])

        self.init_parameters()

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, td: TensorDict, final_h: Tensor, final_p: Tensor) -> Tensor:
        """Compute the logits of the removing a node pair from the current solution

        Args:
            td: TensorDict with the current environment state
            final_h: final node embeddings
            final_p: final positional embeddings
        """

        selection_recent = torch.cat(
            (td["action_record"][:, -3:], td["action_record"].mean(1, True)), 1
        )
        solution = td["rec_current"]

        pre = solution.argsort()  # pre=[1,2,0]
        post = solution.gather(
            1, solution
        )  # post=[1,2,0] # the second neighbour works better
        batch_size, graph_size_plus1, input_dim = final_h.size()

        hflat = final_h.contiguous().view(-1, input_dim)  #################   reshape

        shp = (self.n_heads, batch_size, graph_size_plus1, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, graph_size+1, key_size)
        hidden_Q = torch.matmul(hflat, self.W_Q).view(shp)
        hidden_K = torch.matmul(hflat, self.W_K).view(shp)

        Q_pre = hidden_Q.gather(
            2, pre.view(1, batch_size, graph_size_plus1, 1).expand_as(hidden_Q)
        )
        K_post = hidden_K.gather(
            2, post.view(1, batch_size, graph_size_plus1, 1).expand_as(hidden_Q)
        )

        compatibility = (
            (Q_pre * hidden_K).sum(-1)
            + (hidden_Q * K_post).sum(-1)
            - (Q_pre * K_post).sum(-1)
        )[
            :, :, 1:
        ]  # (n_heads, batch_size, graph_size) (12)

        compatibility_pairing = torch.cat(
            (
                compatibility[:, :, : graph_size_plus1 // 2],
                compatibility[:, :, graph_size_plus1 // 2 :],
            ),
            0,
        )  # (n_heads*2, batch_size, graph_size/2)

        compatibility_pairing = self.agg(
            torch.cat(
                (
                    compatibility_pairing.permute(1, 2, 0),
                    selection_recent.permute(0, 2, 1),
                ),
                -1,
            )
        ).squeeze()  # (batch_size, graph_size/2)

        return compatibility_pairing


class NodePairReinsertionDecoder(ImprovementDecoder):
    """
    N2S Node-Pair Reinsertion decoder based on Ma et al. (2022)
    Given the environment state, the node embeddings (positional embeddings are discarded), and the removed node from the NodePairRemovalDecoder,
    compute the logits for finding places to re-insert the removed pair of pickup and delivery nodes to form a new solution


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = embed_dim
        self.n_heads = num_heads
        self.hidden_dim = embed_dim

        assert embed_dim % num_heads == 0

        self.compater_insert1 = MultiHeadCompat(
            num_heads, embed_dim, embed_dim, embed_dim, embed_dim
        )

        self.compater_insert2 = MultiHeadCompat(
            num_heads, embed_dim, embed_dim, embed_dim, embed_dim
        )

        self.agg = MLP(input_dim=4 * self.n_heads, output_dim=1, num_neurons=[32, 32])

    def forward(self, td: TensorDict, final_h: Tensor, final_p: Tensor) -> torch.Tensor:
        action_removal = td["action"]
        solution = td["rec_current"]

        pos_pickup = (1 + action_removal).view(-1)
        pos_delivery = pos_pickup + solution.size(-1) // 2

        batch_size, graph_size_plus1, input_dim = final_h.size()
        shp = (batch_size, graph_size_plus1, graph_size_plus1, self.n_heads)
        shp_p = (batch_size, -1, 1, self.n_heads)
        shp_d = (batch_size, 1, -1, self.n_heads)

        arange = torch.arange(batch_size, device=final_h.device)
        h_pickup = final_h[arange, pos_pickup].unsqueeze(1)  # (batch_size, 1, input_dim)
        h_delivery = final_h[arange, pos_delivery].unsqueeze(
            1
        )  # (batch_size, 1, input_dim)
        h_K_neibour = final_h.gather(
            1, solution.view(batch_size, graph_size_plus1, 1).expand_as(final_h)
        )  # (batch_size, graph_size+1, input_dim)

        compatibility_pickup_pre = (
            self.compater_insert1(
                h_pickup, final_h
            )  # (n_heads, batch_size, 1, graph_size+1)
            .permute(1, 2, 3, 0)  # (batch_size, 1, graph_size+1, n_heads)
            .view(shp_p)  # (batch_size, graph_size+1, 1, n_heads)
            .expand(shp)  # (batch_size, graph_size+1, graph_size+1, n_heads)
        )
        compatibility_pickup_post = (
            self.compater_insert2(h_pickup, h_K_neibour)
            .permute(1, 2, 3, 0)
            .view(shp_p)
            .expand(shp)
        )
        compatibility_delivery_pre = (
            self.compater_insert1(
                h_delivery, final_h
            )  # (n_heads, batch_size, 1, graph_size+1)
            .permute(1, 2, 3, 0)  # (batch_size, 1, graph_size+1, n_heads)
            .view(shp_d)  # (batch_size, 1, graph_size+1, n_heads)
            .expand(shp)  # (batch_size, graph_size+1, graph_size+1, n_heads)
        )
        compatibility_delivery_post = (
            self.compater_insert2(h_delivery, h_K_neibour)
            .permute(1, 2, 3, 0)
            .view(shp_d)
            .expand(shp)
        )

        compatibility = self.agg(
            torch.cat(
                (
                    compatibility_pickup_pre,
                    compatibility_pickup_post,
                    compatibility_delivery_pre,
                    compatibility_delivery_post,
                ),
                -1,
            )
        ).squeeze()

        return compatibility  # (batch_size, graph_size+1, graph_size+1)


class CriticDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout_rate=0.01) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.project_graph = nn.Linear(self.input_dim, self.input_dim // 2)
        self.project_node = nn.Linear(self.input_dim, self.input_dim // 2)

        self.MLP = MLP(
            input_dim=input_dim + 1,
            output_dim=1,
            num_neurons=[input_dim, input_dim // 2],
            dropout_probs=[dropout_rate, 0.0],
        )

    def forward(self, x: torch.Tensor, best_cost: torch.Tensor) -> torch.Tensor:
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

        fusion_feature = torch.cat(
            (
                fusion.mean(1),
                fusion.max(1)[0],  # max_pooling
                best_cost.to(x.device),
            ),
            -1,
        )  # (batch_size, input_dim + 1)

        value = self.MLP(fusion_feature)

        return value
