from typing import Optional
from tensordict import TensorDict
import torch.nn as nn

from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder


class GFACSEncoder(NARGNNEncoder):
    """
    NARGNNEncoder with log-partition function estimation for training with
    Trajectory Balance (TB) loss (Malkin et al., https://arxiv.org/abs/2201.13259)
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
        z_out_dim: int = 1,
    ):
        super().__init__(
            embed_dim=embed_dim,
            env_name=env_name,
            init_embedding=init_embedding,
            edge_embedding=edge_embedding,
            graph_network=graph_network,
            heatmap_generator=heatmap_generator,
            num_layers_heatmap_generator=num_layers_heatmap_generator,
            num_layers_graph_encoder=num_layers_graph_encoder,
            act_fn=act_fn,
            agg_fn=agg_fn,
            linear_bias=linear_bias,
            k_sparse=k_sparse,
        )
        self.Z_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, z_out_dim)
        )
        self.z_out_dim = z_out_dim

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

        logZ = self.Z_net(graph.edge_attr).reshape(-1, len(td), self.z_out_dim).mean(0)

        # Generate heatmap logits
        heatmap_logits = self.heatmap_generator(graph)

        # Return latent representation (i.e. heatmap logits), initial embeddings and logZ
        return heatmap_logits, node_embed, logZ
