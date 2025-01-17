import torch
import torch.nn as nn

from torch import Tensor

from rl4co.models.common.improvement.base import ImprovementDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RDSDecoder(ImprovementDecoder):
    """
    RDS Decoder for flexible k-opt based on Ma et al. (2023)
    Given the environment state and the node embeddings (positional embeddings are discarded), compute the logits for
    selecting a k-opt exchange on basis moves (S-move, I-move, E-move) from the current solution

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.linear_K1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K3 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K4 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.linear_Q1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q3 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q4 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.linear_V1 = nn.Parameter(torch.Tensor(self.embed_dim))
        self.linear_V2 = nn.Parameter(torch.Tensor(self.embed_dim))

        self.rnn1 = nn.GRUCell(self.embed_dim, self.embed_dim)
        self.rnn2 = nn.GRUCell(self.embed_dim, self.embed_dim)

    def forward(self, h, q1, q2, input_q1, input_q2) -> Tensor:
        bs = h.size(0)

        # GRUs
        q1 = self.rnn1(input_q1, q1)
        q2 = self.rnn2(input_q2, q2)

        # Dual-Stream Attention
        linear_V1 = self.linear_V1.view(1, -1).expand(bs, -1)
        linear_V2 = self.linear_V2.view(1, -1).expand(bs, -1)
        result = (
            linear_V1.unsqueeze(1)
            * torch.tanh(
                self.linear_K1(h)
                + self.linear_Q1(q1).unsqueeze(1)
                + self.linear_K3(h) * self.linear_Q3(q1).unsqueeze(1)
            )
        ).sum(
            -1
        )  # \mu stream
        result += (
            linear_V2.unsqueeze(1)
            * torch.tanh(
                self.linear_K2(h)
                + self.linear_Q2(q2).unsqueeze(1)
                + self.linear_K4(h) * self.linear_Q4(q2).unsqueeze(1)
            )
        ).sum(
            -1
        )  # \lambda stream

        return result, q1, q2
