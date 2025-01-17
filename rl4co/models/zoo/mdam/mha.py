import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiHeadAttentionMDAM(nn.Module):
    def __init__(self, embed_dim, n_heads, last_one=False, sdpa_fn=None):
        super(MultiHeadAttentionMDAM, self).__init__()

        if sdpa_fn is not None:
            log.warning("sdpa_fn is not used in this implementation")

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.norm_factor = 1 / math.sqrt(embed_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))

        self.init_parameters()
        self.last_one = last_one

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.embed_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = float("-inf")

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.embed_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)
        if self.last_one:
            return (out, attn, V)
        return out
