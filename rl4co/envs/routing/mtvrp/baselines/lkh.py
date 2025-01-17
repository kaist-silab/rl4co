import lkh
import numpy as np

from tensordict import TensorDict
from torch import Tensor

from .constants import LKH_SCALING_FACTOR, ROUTEFINDER2LKH
from .utils import scale


def solve(
    instance: TensorDict,
    max_runtime: float,
    problem_type: str,
    num_runs: int,
    solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an AnyVRP instance with OR-Tools.

    Args:
        instance: The AnyVRP instance to solve.
        max_runtime: The maximum runtime for the solver.
        problem_type: The problem type for LKH3.
        num_runs: The number of runs to perform and returns the best result.
        solver_loc: The location of the LKH3 solver executable.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    problem = instance2problem(instance, problem_type, LKH_SCALING_FACTOR)
    action, cost = _solve(problem, max_runtime, num_runs, solver_loc)
    cost /= -LKH_SCALING_FACTOR

    return action, cost


def _solve(
    problem: lkh.LKHProblem,
    max_runtime: float,
    num_runs: int,
    solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an instance with LKH3.

    Args:
        problem: The LKHProblem instance.
        max_runtime: The maximum runtime for each solver run.
        num_runs: The number of runs to perform and returns the best result.
        solver_loc: The location of the LKH3 solver executable.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    routes, cost = lkh.solve(
        solver_loc,
        problem=problem,
        time_limit=max_runtime,
        runs=num_runs,
    )

    action = routes2action(routes)
    return action, cost


def instance2problem(
    instance: TensorDict,
    problem_type: str,
    scaling_factor,
) -> lkh.LKHProblem:
    """
    Converts an AnyVRP instance to an LKHProblem instance.

    Args:
        instance: The AnyVRP instance to convert.
        problem_type: The problem type for LKH3.
        scaling_factor: The scaling factor to apply to the instance data.

    Returns:
        The LKHProblem instance.
    """
    num_locations = instance["demand_linehaul"].size()[0]

    # Data specifications
    specs = {}
    specs["DIMENSION"] = num_locations
    specs["CAPACITY"] = scale(instance["vehicle_capacity"], scaling_factor)

    if not np.isinf(distance_limit := instance["distance_limit"]).any():
        specs["DISTANCE"] = scale(distance_limit, scaling_factor)

    specs["EDGE_WEIGHT_TYPE"] = "EXPLICIT"
    specs["EDGE_WEIGHT_FORMAT"] = "FULL_MATRIX"
    specs["NODE_COORD_TYPE"] = "TWOD_COORDS"

    # LKH can only solve VRP variants that are explicitly supported (so no
    # arbitrary combinations between individual supported features). We can
    # support some open variants with some modeling tricks.
    lkh_problem_type = ROUTEFINDER2LKH[problem_type]
    if lkh_problem_type is None:
        raise ValueError(f"Problem type {problem_type} is not supported by LKH.")

    specs["TYPE"] = lkh_problem_type

    # Weird LKH quirk: specifying the number of vehicles lets (D)CVRP hang.
    if lkh_problem_type not in ["CVRP", "DCVRP"]:
        specs["VEHICLES"] = num_locations - 1

    # Data sections
    sections = {}
    sections["NODE_COORD_SECTION"] = scale(instance["locs"], scaling_factor)

    demand_linehaul = scale(instance["demand_linehaul"], scaling_factor)
    demand_backhaul = scale(instance["demand_backhaul"], scaling_factor)
    sections["DEMAND_SECTION"] = demand_linehaul + demand_backhaul

    time_windows = scale(instance["time_windows"], scaling_factor)
    sections["TIME_WINDOW_SECTION"] = time_windows

    service_times = scale(instance["durations"], scaling_factor)
    sections["SERVICE_TIME_SECTION"] = service_times

    distances = instance["cost_matrix"]
    backhaul_class = instance["backhaul_class"]

    if backhaul_class == 1:
        # VRPB has a backhaul section that specifies the backhaul nodes.
        backhaul_idcs = np.flatnonzero(instance["demand_backhaul"]).tolist()
        sections["BACKHAUL_SECTION"] = backhaul_idcs + [-1]

        # linehaul = np.flatnonzero(demand_linehaul > 0)
        # backhaul = np.flatnonzero(demand_backhaul > 0)
        # distances[np.ix_(backhaul, linehaul)] = time_windows.max()

    elif backhaul_class == 2:
        # VRPMPD has a pickup and delivery section that specifies the pickup
        # and delivery quantities for each node, as well as the time windows.
        # The regular time window section is redundant in this case.
        data = [
            [
                0,  # dummy
                time_windows[idx][0],
                time_windows[idx][1],
                service_times[idx],
                demand_backhaul[idx],
                demand_linehaul[idx],
            ]
            for idx in range(num_locations)
        ]
        sections["PICKUP_AND_DELIVERY_SECTION"] = data

    if instance["open_route"]:
        # Arcs to the depot are set to zero as vehicles don’t need to return.
        distances[:, 0] = 0

    sections["EDGE_WEIGHT_SECTION"] = scale(distances, scaling_factor)

    # Convert to VRPLIB-like string.
    problem = "\n".join(f"{k} : {v}" for k, v in specs.items())
    problem += "\n" + "\n".join(_format(name, data) for name, data in sections.items())
    problem += "\n" + "\n".join(["DEPOT_SECTION", "1", "-1", "EOF"])

    return lkh.LKHProblem.parse(problem)


def _is_1D(data) -> bool:
    for elt in data:
        if isinstance(elt, (list, tuple, np.ndarray)):
            return False
    return True


def _format(name: str, data) -> str:
    """
    Formats a data section.

    Args:
        name: The name of the section.
        data: The data to be formatted.

    Returns:
        A VRPLIB-formatted data section.
    """
    section = [name]
    include_idx = name not in ["EDGE_WEIGHT_SECTION", "BACKHAUL_SECTION"]

    if name == "BACKHAUL_SECTION":
        # Treat backhaul section as row vector.
        section.append("\t".join(str(val) for val in data))

    elif _is_1D(data):
        # Treat 1D arrays as column vectors, so each element is a row.
        for idx, elt in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            section.append(prefix + str(elt))
    else:
        for idx, row in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            rest = "\t".join([str(elt) for elt in row])
            section.append(prefix + rest)

    return "\n".join(section)


def routes2action(routes: list[list[int]]) -> list[int]:
    """
    Converts LKH routes to an action.
    """
    # LKH routes are location-indexed, which in turn are 1-indexed. The first
    # location is always the depot, so we subtract 2 to get client indices.
    # LKH routes are 1-indexed, so we subtract 1 to get client indices.
    routes_ = [[client - 1 for client in route] for route in routes]
    return [visit for route in routes_ for visit in route + [0]]
