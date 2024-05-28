import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": StaticEmbedding,
        "atsp": StaticEmbedding,
        "cvrp": StaticEmbedding,
        "cvrptw": StaticEmbedding,
        "ffsp": StaticEmbedding,
        "svrp": StaticEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": StaticEmbedding,
        "spctsp": StaticEmbedding,
        "op": StaticEmbedding,
        "dpp": StaticEmbedding,
        "mdpp": StaticEmbedding,
        "pdp": StaticEmbedding,
        "mtsp": StaticEmbedding,
        "smtwtp": StaticEmbedding,
        "shpp": StaticEmbedding,
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, StaticEmbedding)(**config)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class SDVRPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Split Delivery Vehicle Routing Problem (SDVRP).
    Embed the following node features to the embedding space:
        - demand_with_depot: demand of the customers and the depot
    The demand with depot is used to modify the query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        demands_with_depot = td["demand_with_depot"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
