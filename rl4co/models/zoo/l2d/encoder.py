from rl4co.models.nn.env_embeddings.init import JSSPInitEmbedding
from rl4co.models.nn.graph.gcn import GCNEncoder
from rl4co.utils.ops import adj_to_pyg_edge_index


class GCN4JSSP(GCNEncoder):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        init_embedding=None,
        **init_embedding_kwargs,
    ):
        def edge_idx_fn(td, _):
            return adj_to_pyg_edge_index(td["adjacency"])

        if init_embedding is None:
            init_embedding = JSSPInitEmbedding(embed_dim, **init_embedding_kwargs)

        super().__init__(
            env_name="jssp",
            embed_dim=embed_dim,
            num_layers=num_layers,
            edge_idx_fn=edge_idx_fn,
            init_embedding=init_embedding,
        )
