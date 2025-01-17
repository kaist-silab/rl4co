import math

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from rl4co.models.common import ImprovementEncoder
from rl4co.models.nn.ops import AdaptiveSequential, Normalization
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# implements the Multi-head DAC-Att module
class DAC_ATT(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(DAC_ATT, self).__init__()

        self.n_heads = n_heads

        self.key_dim = self.val_dim = embed_dim // n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.norm_factor = 1 / math.sqrt(1 * self.key_dim)

        # W_h^Q in the paper
        self.W_query_node = nn.Parameter(
            torch.Tensor(n_heads, self.input_dim, self.key_dim)
        )
        # W_g^Q in the paper
        self.W_query_pos = nn.Parameter(
            torch.Tensor(n_heads, self.input_dim, self.key_dim)
        )
        # W_h^K in the paper
        self.W_key_node = nn.Parameter(
            torch.Tensor(n_heads, self.input_dim, self.key_dim)
        )
        # W_g^K in the paper
        self.W_key_pos = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))

        # W_h^V and W_h^Vref in the paper
        self.W_val_node = nn.Parameter(
            torch.Tensor(2 * n_heads, self.input_dim, self.val_dim)
        )
        # W_g^V and W_g^Vref in the paper
        self.W_val_pos = nn.Parameter(
            torch.Tensor(2 * n_heads, self.input_dim, self.val_dim)
        )

        # W_h^O and W_g^O in the paper
        if embed_dim is not None:
            self.W_out_node = nn.Parameter(
                torch.Tensor(n_heads, 2 * self.key_dim, embed_dim)
            )
            self.W_out_pos = nn.Parameter(
                torch.Tensor(n_heads, 2 * self.key_dim, embed_dim)
            )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_node_in, h_pos_in):  # input (NFEs, PFEs)
        # h,g should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h_node_in.size()

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_v = (2, self.n_heads, batch_size, graph_size, -1)

        h_node = h_node_in.contiguous().view(-1, input_dim)
        h_pos = h_pos_in.contiguous().view(-1, input_dim)

        Q_node = torch.matmul(h_node, self.W_query_node).view(shp)
        Q_pos = torch.matmul(h_pos, self.W_query_pos).view(shp)

        K_node = torch.matmul(h_node, self.W_key_node).view(shp)
        K_pos = torch.matmul(h_pos, self.W_key_pos).view(shp)

        V_node = torch.matmul(h_node, self.W_val_node).view(shp_v)
        V_pos = torch.matmul(h_pos, self.W_val_pos).view(shp_v)

        # Get attention correlations and norm by softmax
        node_correlations = self.norm_factor * torch.matmul(
            Q_node, K_node.transpose(2, 3)
        )
        pos_correlations = self.norm_factor * torch.matmul(Q_pos, K_pos.transpose(2, 3))
        attn1 = F.softmax(node_correlations, dim=-1)  # head, bs, n, n
        attn2 = F.softmax(pos_correlations, dim=-1)  # head, bs, n, n

        heads_node_1 = torch.matmul(attn1, V_node[0])  # self-attn
        heads_node_2 = torch.matmul(attn2, V_node[1])  # cross-aspect ref attn

        heads_pos_1 = torch.matmul(attn1, V_pos[0])  # cross-aspect ref attn
        heads_pos_2 = torch.matmul(attn2, V_pos[1])  # self-attn

        heads_node = torch.cat((heads_node_1, heads_node_2), -1)
        heads_pos = torch.cat((heads_pos_1, heads_pos_2), -1)

        # get output
        out_node = torch.mm(
            heads_node.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_node.view(-1, self.embed_dim),
        ).view(batch_size, graph_size, self.embed_dim)

        out_pos = torch.mm(
            heads_pos.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_pos.view(-1, self.embed_dim),
        ).view(batch_size, graph_size, self.embed_dim)

        return out_node, out_pos  # dual-aspect representation (NFEs, PFEs)


# implements the DAC encoder
class DACTEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization="layer",
    ):
        super(DACTEncoderLayer, self).__init__()

        self.MHA_sublayer = DACsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFNsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1, out2)


# implements the DAC encoder (DAC-Att sublayer)
class DACsubLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization="layer",
    ):
        super(DACsubLayer, self).__init__()

        self.MHA = DAC_ATT(n_heads, input_dim=embed_dim, embed_dim=embed_dim)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)

        # Normalization
        return self.Norm(out1 + input1), self.Norm(out2 + input2)


# implements the DAC encoder (FFN sublayer)
class FFNsubLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization="layer",
    ):
        super(FFNsubLayer, self).__init__()

        self.FF1 = (
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )

        self.FF2 = (
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input1, input2):
        # FF and Residual connection
        out1 = self.FF1(input1)
        out2 = self.FF2(input2)

        # Normalization
        return self.Norm(out1 + input1), self.Norm(out2 + input2)


class DACTEncoder(ImprovementEncoder):
    """Dual-Aspect Collaborative Transformer Encoder as in Ma et al. (2021)

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the node embeddings
        pos_embedding: Module to use for the initialization of the positional embeddings
        env_name: Name of the environment used to initialize embeddings
        pos_type: Name of the used positional encoding method (CPE or APE)
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
    """

    def __init__(
        self,
        embed_dim: int = 64,
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        env_name: str = "tsp_kopt",
        pos_type: str = "CPE",
        num_heads: int = 4,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 64,
    ):
        super(DACTEncoder, self).__init__(
            embed_dim=embed_dim,
            env_name=env_name,
            pos_type=pos_type,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        assert self.env_name in ["tsp_kopt"], NotImplementedError()

        self.net = AdaptiveSequential(
            *(
                DACTEncoderLayer(
                    num_heads,
                    embed_dim,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def _encoder_forward(self, init_h: Tensor, init_p: Tensor) -> Tuple[Tensor, Tensor]:
        NFE, PFE = self.net(init_h, init_p)

        return NFE, PFE
