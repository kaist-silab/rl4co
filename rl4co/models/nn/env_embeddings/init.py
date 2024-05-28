import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
from rl4co.models.nn.ops import PositionalEncoding

def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPInitEmbedding,
        "atsp": TSPInitEmbedding,
        "matnet": MatNetInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "cvrptw": VRPTWInitEmbedding,
        "svrp": SVRPInitEmbedding,
        "sdvrp": VRPInitEmbedding,
        "pctsp": PCTSPInitEmbedding,
        "spctsp": PCTSPInitEmbedding,
        "op": OPInitEmbedding,
        "dpp": DPPInitEmbedding,
        "mdpp": MDPPInitEmbedding,
        "pdp": PDPInitEmbedding,
        "pdp_ruin_repair": TSPInitEmbedding,
        "mtsp": MTSPInitEmbedding,
        "smtwtp": SMTWTPInitEmbedding,
        "mdcpdp": MDCPDPInitEmbedding,
        "fjsp": FJSPFeatureEmbedding,
        "mtvrp":MTVRPInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class MatNetInitEmbedding(nn.Module):
    """
    Preparing the initial row and column embeddings for FFSP.

    Reference:
    https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L51


    """

    def __init__(self, embed_dim: int, mode: str = "RandomOneHot") -> None:
        super().__init__()

        self.embed_dim = embed_dim
        assert mode in {
            "RandomOneHot",
            "Random",
        }, "mode must be one of ['RandomOneHot', 'Random']"
        self.mode = mode

    def forward(self, td: TensorDict):
        dmat = td["cost_matrix"]
        b, r, c = dmat.shape

        row_emb = torch.zeros(b, r, self.embed_dim, device=dmat.device)

        if self.mode == "RandomOneHot":
            # MatNet uses one-hot encoding for column embeddings
            # https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L60
            col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device)
            rand = torch.rand(b, c)
            rand_idx = rand.argsort(dim=1)
            b_idx = torch.arange(b)[:, None].expand(b, c)
            n_idx = torch.arange(c)[None, :].expand(b, c)
            col_emb[b_idx, n_idx, rand_idx] = 1.0

        elif self.mode == "Random":
            col_emb = torch.rand(b, r, self.embed_dim, device=dmat.device)
        else:
            raise NotImplementedError

        return row_emb, col_emb, dmat


class VRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(VRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class VRPTWInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 6):
        # node_dim = 6: x, y, demand, tw start, tw end, service time
        super(VRPTWInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        durations = td["durations"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (cities, td["demand"][..., None], time_windows, durations[..., None]), -1
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)
    

class MTVRPInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 5):
        # node_dim = 5: x, y, demand, tw start, tw end
        super(MTVRPInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        #durations = td["durations"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # embeddings
        demands =  td["demand_linehaul"][..., None] - td["demand_backhaul"][..., None]

        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (cities, demands[:,1:], time_windows), -1
            )
        )

        return torch.cat((depot_embedding, node_embeddings), -2)


class SVRPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(SVRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, skill
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(torch.cat((cities, td["skills"]), -1))
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class PCTSPInitEmbedding(nn.Module):
    """Initial embedding for the Prize Collecting Traveling Salesman Problems (PCTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - expected_prize: expected prize for visiting the customers.
            In PCTSP, this is the actual prize. In SPCTSP, this is the expected prize.
        - penalty: penalty for not visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["expected_prize"][..., None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_city+1, embed_dim
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class OPInitEmbedding(nn.Module):
    """Initial embedding for the Orienteering Problems (OP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - prize: prize for visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(OPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, prize
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["prize"][..., 1:, None],  # exclude depot
                ),
                -1,
            )
        )
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class DPPInitEmbedding(nn.Module):
    """Initial embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: index of the (single) probe cell. We embed the euclidean distance from the probe to all cells.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(DPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim // 2, linear_bias)  # locs
        self.init_embed_probe = nn.Linear(1, embed_dim // 2, linear_bias)  # probe

    def forward(self, td):
        node_embeddings = self.init_embed(td["locs"])
        probe_embedding = self.init_embed_probe(
            self._distance_probe(td["locs"], td["probe"])
        )
        return torch.cat([node_embeddings, probe_embedding], -1)

    def _distance_probe(self, locs, probe):
        # Euclidean distance from probe to all locations
        probe_loc = torch.gather(locs, 1, probe.unsqueeze(-1).expand(-1, -1, 2))
        return torch.norm(locs - probe_loc, dim=-1).unsqueeze(-1)


class MDPPInitEmbedding(nn.Module):
    """Initial embedding for the Multi-port Placement Problem (MDPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: indexes of the probe cells (multiple). We embed the euclidean distance of each cell to the closest probe.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)  # locs
        self.init_embed_probe_distance = nn.Linear(
            1, embed_dim, linear_bias
        )  # probe_distance
        self.project_out = nn.Linear(embed_dim * 2, embed_dim, linear_bias)

    def forward(self, td):
        probes = td["probe"]
        locs = td["locs"]
        node_embeddings = self.init_embed(locs)

        # Get the shortest distance from any probe
        dist = torch.cdist(locs, locs, p=2)
        dist[~probes] = float("inf")
        min_dist, _ = torch.min(dist, dim=1)
        min_probe_dist_embedding = self.init_embed_probe_distance(min_dist[..., None])

        return self.project_out(
            torch.cat([node_embeddings, min_probe_dist_embedding], -1)
        )


class PDPInitEmbedding(nn.Module):
    """Initial embedding for the Pickup and Delivery Problem (PDP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class MTSPInitEmbedding(nn.Module):
    """Initial embedding for the Multiple Traveling Salesman Problem (mTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, cities)
    """

    def __init__(self, embed_dim, linear_bias=True):
        """NOTE: new made by Fede. May need to be checked"""
        super(MTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot_embedding = self.init_embed_depot(td["locs"][..., 0:1, :])
        node_embedding = self.init_embed(td["locs"][..., 1:, :])
        return torch.cat([depot_embedding, node_embedding], -2)


class SMTWTPInitEmbedding(nn.Module):
    """Initial embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Embed the following node features to the embedding space:
        - job_due_time: due time of the jobs
        - job_weight: weights of the jobs
        - job_process_time: the processing time of jobs
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SMTWTPInitEmbedding, self).__init__()
        node_dim = 3  # job_due_time, job_weight, job_process_time
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        job_due_time = td["job_due_time"]
        job_weight = td["job_weight"]
        job_process_time = td["job_process_time"]
        feat = torch.stack((job_due_time, job_weight, job_process_time), dim=-1)
        out = self.init_embed(feat)
        return out


class MDCPDPInitEmbedding(nn.Module):
    """Initial embedding for the MDCPDP environment
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDCPDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        num_depots = td["capacity"].size(-1)
        depot, locs = td["locs"][..., 0:num_depots, :], td["locs"][..., num_depots:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)

class FJSPFeatureEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, norm_coef: int = 100):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_coef = norm_coef

        self.init_ope_embed = nn.Linear(4, self.embed_dim, bias=False)
        self.edge_embed = nn.Linear(1, embed_dim, bias=False)

        self.ope_pos_enc = PositionalEncoding(embed_dim)
        # TODO allow for reencoding after each step
        self.stepwise = False

    def forward(self, td: TensorDict):
        if self.stepwise:
            ops_emb = self._stepwise_operations_embed(td)
            ma_emb = self._stepwise_machine_embed(td)
            edge_emb = None
        else:
            ops_emb = self._init_operations_embed(td)
            ma_emb = self._init_machine_embed(td)
            edge_emb = self._init_edge_embed(td)
        return ma_emb, ops_emb, edge_emb

    def _init_operations_embed(self, td: TensorDict):
        pos = td["ops_sequence_order"]

        features = [
            td["lbs"].unsqueeze(-1) / self.norm_coef,
            td["is_ready"].unsqueeze(-1),
            td["num_eligible"].unsqueeze(-1),
            td["ops_job_map"].unsqueeze(-1),
        ]
        features = torch.cat(features, dim=-1)
        # (bs, num_ops, emb_dim)
        ops_embeddings = self.init_ope_embed(features)

        # (bs, num_ops, emb_dim)
        ops_embeddings = self.ope_pos_enc(ops_embeddings, pos.to(torch.int64))
        # zero out padded entries
        ops_embeddings[td["pad_mask"].unsqueeze(-1).expand_as(ops_embeddings)] = 0
        return ops_embeddings

    def _init_machine_embed(self, td: TensorDict):
        bs, num_ma = td["busy_until"].shape
        ma_embeddings = torch.zeros(
            (bs, num_ma, self.embed_dim), device=td.device, dtype=torch.float32
        )
        return ma_embeddings

    def _init_edge_embed(self, td: TensorDict):
        proc_times = td["proc_times"].unsqueeze(-1) / self.norm_coef
        edge_embed = self.edge_embed(proc_times)
        return edge_embed

    def _stepwise_operations_embed(self, td: TensorDict):
        raise NotImplementedError("Stepwise encoding not yet implemented")

    def _stepwise_machine_embed(self, td: TensorDict):
        raise NotImplementedError("Stepwise encoding not yet implemented")