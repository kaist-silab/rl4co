from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import torch

from pyvrp import (
    Client,
    CostEvaluator,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Solution,
    VehicleType,
)
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
)
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import get_distance_matrix
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


C = (
    10**4
)  # Scaling factor for the data, to convert the float values to integers as required by PyVRP


def local_search(
    td: TensorDict,
    actions: torch.Tensor,
    max_trials: int = 10,
    neighbourhood_params: dict | None = None,
    load_penalty: float = 0.2,
    allow_infeasible_solution: bool = False,
    seed: int = 0,
    num_workers: int = 1,
):
    """
    Improve the solution using local search for CVRP, based on PyVRP.

    Args:
        td: TensorDict, td from env with shape [batch_size,]
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
        max_trials: int, maximum number of trials for local search
        neighbourhood_params: dict, parameters for neighbourhood search
        load_penalty: int, penalty for exceeding the vehicle capacity
        allow_infeasible_solution: bool, whether to allow infeasible solutions
        seed: int, random seed for local search
        num_workers: int, number of workers for parallel processing
    Returns:
        torch.Tensor, Improved tour indices with shape [batch_size, max_seq_len]
    """

    # Convert tensors to numpy arrays
    # Note: to avoid the overhead of device transfer, we recommend to pass the tensors in cpu
    actions_np = actions.detach().cpu().numpy()
    positions_np = td["locs"].detach().cpu().numpy()  # [batch_size, num_loc + 1, 2]
    demands_np = td["demand"].detach().cpu().numpy()  # [batch_size, num_loc]
    demands_np = np.pad(demands_np, ((0, 0), (1, 0)), mode="constant")  # Add depot demand
    distances = td.get("distances", None)  # [batch_size, num_loc + 1, num_loc + 1]
    if distances is None:
        distances_np = get_distance_matrix(td["locs"]).numpy()
    else:
        distances_np = distances.detach().cpu().numpy()

    max_trials = 1 if allow_infeasible_solution else max_trials

    partial_func = partial(
        local_search_single,
        neighbourhood_params=neighbourhood_params,
        load_penalty=load_penalty,
        allow_infeasible_solution=allow_infeasible_solution,
        max_trials=max_trials,
        seed=seed,
    )

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            new_actions = pool.starmap(
                partial_func, zip(actions_np, positions_np, demands_np, distances_np)
            )
    else:
        new_actions = [
            partial_func(*args)
            for args in zip(actions_np, positions_np, demands_np, distances_np)
        ]

    # padding with zero
    lengths = [len(act) for act in new_actions]
    max_length = max(lengths)
    new_actions = np.array(
        [
            np.pad(act, (0, max_length - length), mode="constant")
            for act, length in zip(new_actions, lengths)
        ]
    )
    return torch.from_numpy(new_actions[:, :-1].astype(np.int64)).to(
        td.device
    )  # We can remove the last zero


def local_search_single(
    path: np.ndarray,
    positions: np.ndarray,
    demands: np.ndarray,
    distances: np.ndarray,
    neighbourhood_params: dict | None = None,
    allow_infeasible_solution: bool = False,
    load_penalty: float = 0.2,
    max_trials: int = 10,
    seed: int = 0,
) -> np.ndarray:
    data = make_data(positions, demands, distances)
    solution = make_solution(data, path)
    ls_operator = make_search_operator(data, seed, neighbourhood_params)

    improved_solution, is_feasible = perform_local_search(
        ls_operator,
        solution,
        int(load_penalty * C),  # * C as we scale the data in `make_data`
        remaining_trials=max_trials,
    )

    # Return the original path if no feasible solution is found
    if not is_feasible and not allow_infeasible_solution:
        return path

    # Recover the path from the sub-routes in the solution
    route_list = [
        idx for route in improved_solution.routes() for idx in [0] + route.visits()
    ] + [0]
    return np.array(route_list)


def make_data(
    positions: np.ndarray, demands: np.ndarray, distances: np.ndarray
) -> ProblemData:
    positions = (positions * C).astype(int)
    distances = (distances * C).astype(int)

    capacity = C
    demands = np.round(demands * capacity).astype(int)

    return ProblemData(
        clients=[
            Client(x=pos[0], y=pos[1], delivery=d)
            for pos, d in zip(positions[1:], demands[1:])
        ],
        depots=[Depot(x=positions[0][0], y=positions[0][1])],
        vehicle_types=[
            VehicleType(
                len(positions) - 1,
                capacity,
                0,
                name=",".join(map(str, range(1, len(positions)))),
            )
        ],
        distance_matrices=[distances],
        duration_matrices=[np.zeros_like(distances)],
    )


def make_solution(data: ProblemData, path: np.ndarray) -> Solution:
    # Split the paths into sub-routes by the zeros
    routes = [
        arr[1:].tolist() for arr in np.split(path, np.where(path == 0)[0]) if len(arr) > 1
    ]
    return Solution(data, routes)


def make_search_operator(
    data: ProblemData, seed=0, neighbourhood_params: dict | None = None
) -> LocalSearch:
    rng = RandomNumberGenerator(seed)
    neighbours = compute_neighbours(
        data, NeighbourhoodParams(**(neighbourhood_params or {}))
    )
    ls = LocalSearch(data, rng, neighbours)
    for node_op in NODE_OPERATORS:
        ls.add_node_operator(node_op(data))
    for route_op in ROUTE_OPERATORS:
        ls.add_route_operator(route_op(data))
    return ls


def perform_local_search(
    ls_operator: LocalSearch,
    solution: Solution,
    load_penalty: int,
    remaining_trials: int = 5,
) -> Tuple[Solution, bool]:
    cost_evaluator = CostEvaluator(
        load_penalty=load_penalty, tw_penalty=0, dist_penalty=0
    )
    improved_solution = ls_operator(solution, cost_evaluator)
    remaining_trials -= 1
    if is_feasible := improved_solution.is_feasible() or remaining_trials == 0:
        return improved_solution, is_feasible

    # print("Warning: Infeasible solution found from local search.",
    #       "This will slow down the search due to the repeated local search runs.")

    # If infeasible, run the local search again with a higher penalty
    return perform_local_search(
        ls_operator, solution, load_penalty * 10, remaining_trials=remaining_trials
    )
