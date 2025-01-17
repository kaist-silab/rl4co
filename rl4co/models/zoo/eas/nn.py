import torch
import torch.nn as nn


class EASLayerNet(nn.Module):
    """Instantiate weights and biases for the added layer.
    The layer is defined as: h = relu(emb * W1 + b1); out = h * W2 + b2.
    Wrapping in `nn.Parameter` makes the parameters trainable and sets gradient to True.

    Args:
        num_instances: Number of instances in the dataset
        emb_dim: Dimension of the embedding
    """

    def __init__(self, num_instances: int, emb_dim: int):
        super().__init__()
        # W2 and b2 are initialized to zero so in the first iteration the layer is identity
        self.W1 = nn.Parameter(torch.randn(num_instances, emb_dim, emb_dim))
        self.b1 = nn.Parameter(torch.randn(num_instances, 1, emb_dim))
        self.W2 = nn.Parameter(torch.zeros(num_instances, emb_dim, emb_dim))
        self.b2 = nn.Parameter(torch.zeros(num_instances, 1, emb_dim))
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.b1)

    def forward(self, *args):
        """emb: [num_instances, group_num, emb_dim]"""
        # get tensor arg (from partial instantiation)
        emb = [arg for arg in args if isinstance(arg, torch.Tensor)][0]
        h = torch.relu(torch.matmul(emb, self.W1) + self.b1.expand_as(emb))
        return torch.matmul(h, self.W2) + self.b2.expand_as(h)
