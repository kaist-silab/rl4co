import numpy as np
import pyvrp as pyvrp

from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict
from torch import Tensor

from .constants import PYVRP_SCALING_FACTOR
from .utils import scale


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instance with PyVRP.

    Args:
        instance: The AnyVRP instance to solve.
        max_runtime: The maximum runtime for the solver.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MaxRuntime(max_runtime)
    result = _solve(data, stop)

    solution = result.best
    action = solution2action(solution)
    cost = -result.cost() / PYVRP_SCALING_FACTOR

    return action, cost


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Converts an AnyVRP instance to a ProblemData instance.

    Args:
        instance: The AnyVRP instance to convert.
        scaling_factor: The scaling factor to use for the conversion.

    Returns:
        The ProblemData instance.
    """
    num_locs = instance["demand_backhaul"].size()[0]

    time_windows = scale(instance["time_windows"], scaling_factor)
    pickup = scale(instance["demand_backhaul"], scaling_factor)
    delivery = scale(instance["demand_linehaul"], scaling_factor)
    service = scale(instance["service_time"], scaling_factor)
    coords = scale(instance["locs"], scaling_factor)
    capacity = scale(instance["vehicle_capacity"], scaling_factor)
    max_distance = scale(instance["distance_limit"], scaling_factor)

    depot = Depot(
        x=coords[0][0],
        y=coords[0][1],
    )

    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            tw_early=time_windows[idx][0],
            tw_late=time_windows[idx][1],
            delivery=delivery[idx],
            pickup=pickup[idx],
            service_duration=service[idx],
        )
        for idx in range(1, num_locs)
    ]

    vehicle_type = VehicleType(
        num_available=num_locs - 1,  # one vehicle per client
        capacity=capacity,
        max_distance=max_distance,
        tw_early=time_windows[0][0],
        tw_late=time_windows[0][1],
    )

    matrix = scale(instance["cost_matrix"], scaling_factor)

    if instance["open_route"]:
        # Vehicles do not need to return to the depot, so we set all arcs
        # to the depot to zero.
        matrix[:, 0] = 0

    if instance["backhaul_class"] == 1:  # VRP with backhauls
        # In VRPB, linehauls must be served before backhauls. This can be
        # enforced by setting a high value for the distance/duration from depot
        # to backhaul (forcing linehaul to be served first) and a large value
        # from backhaul to linehaul (avoiding linehaul after backhaul clients).
        linehaul = np.flatnonzero(delivery > 0)
        backhaul = np.flatnonzero(pickup > 0)
        # Note: we remove the constraint that we cannot visit backhauls *only* in a
        # a single route as per Slack discussion
        # matrix[0, backhaul] = MAX_VALUE
        matrix[np.ix_(backhaul, linehaul)] = MAX_VALUE

    return ProblemData(clients, [depot], [vehicle_type], [matrix], [matrix])


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    """
    return [visit for route in solution.routes() for visit in route.visits() + [0]]
