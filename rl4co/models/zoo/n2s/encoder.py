import math

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from rl4co.models.common import ImprovementEncoder
from rl4co.models.nn.attention import MultiHeadCompat
from rl4co.models.nn.ops import AdaptiveSequential, Normalization
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Synth_Attention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()

        hidden_dim = input_dim // n_heads

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(2 * n_heads, 2 * n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(2 * n_heads, n_heads),
        )

        self.W_out = nn.Parameter(torch.Tensor(n_heads, hidden_dim, input_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h should be (batch_size, n_query, input_dim)
        batch_size, n_query, input_dim = h_fea.size()

        hflat = h_fea.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, n_query, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), aux_att_score), 0)

        attn_raw = compatibility.permute(
            1, 2, 3, 0
        )  # (batch_size, n_query, n_key, n_heads)
        attn = self.score_aggr(attn_raw).permute(
            3, 0, 1, 2
        )  # (n_heads, batch_size, n_query, n_key)
        heads = torch.matmul(
            F.softmax(attn, dim=-1), V
        )  # (n_heads, batch_size, n_query, hidden_dim)

        h_wave = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, self.n_heads * self.hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            self.W_out.view(-1, self.input_dim),  # (n_heads * hidden_dim, input_dim)
        ).view(batch_size, n_query, self.input_dim)

        return h_wave, aux_att_score


class SynthAttNormSubLayer(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.SynthAtt = Synth_Attention(n_heads, input_dim)

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention and Residual connection
        h_wave, aux_att_score = self.SynthAtt(h_fea, aux_att_score)

        # Normalization
        return self.Norm(h_wave + h_fea), aux_att_score


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim, bias=False),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim, bias=False)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class N2SEncoderLayer(nn.Module):
    def __init__(
        self, n_heads: int, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.SynthAttNorm_sublayer = SynthAttNormSubLayer(
            n_heads, input_dim, normalization
        )

        self.FFNorm_sublayer = FFNormSubLayer(
            input_dim, feed_forward_hidden, normalization
        )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_wave, aux_att_score = self.SynthAttNorm_sublayer(h_fea, aux_att_score)
        return self.FFNorm_sublayer(h_wave), aux_att_score


class N2SEncoder(ImprovementEncoder):
    """Neural Neighborhood Search Encoder as in Ma et al. (2022)
    First embed the input and then process it with a Graph AttepdN2ntion Network.

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
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        env_name: str = "pdp_ruin_repair",
        pos_type: str = "CPE",
        num_heads: int = 4,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 128,
    ):
        super(N2SEncoder, self).__init__(
            embed_dim=embed_dim,
            init_embedding=init_embedding,
            pos_embedding=pos_embedding,
            env_name=env_name,
            pos_type=pos_type,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        self.pos_net = MultiHeadCompat(num_heads, embed_dim, feedforward_hidden)

        self.net = AdaptiveSequential(
            *(
                N2SEncoderLayer(
                    num_heads,
                    embed_dim,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def _encoder_forward(self, init_h: Tensor, init_p: Tensor) -> Tuple[Tensor, Tensor]:
        embed_p = self.pos_net(init_p)
        final_h, final_p = self.net(init_h, embed_p)

        return final_h, final_p
