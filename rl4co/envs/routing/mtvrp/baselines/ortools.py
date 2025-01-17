from dataclasses import dataclass
from typing import Optional

import numpy as np
import routefinder.baselines.pyvrp as pyvrp

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tensordict import TensorDict
from torch import Tensor

from .constants import ORTOOLS_SCALING_FACTOR


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves an MTVRP instance with OR-Tools.

    Args:
        instance: The MTVRP instance to solve.
        max_runtime: The maximum runtime for the solver.

    Returns:
        A tuple containing the action and the cost, respectively.

    Note:
        This function depends on PyVRP's data converter to convert the MTVRP
        instance to an OR-Tools compatible format. Future versions should
        implement a direct conversion.
    """
    data = instance2data(instance)
    action, cost = _solve(data, max_runtime)
    cost /= ORTOOLS_SCALING_FACTOR
    cost *= -1

    return action, cost


@dataclass
class ORToolsData:
    """
    Convenient dataclass for instance data when using OR-Tools as solver.

    Args:
        depot: The depot index.
        distance_matrix: The distance matrix between locations.
        duration_matrix: The duration matrix between locations. This includes service times.
        num_vehicles: The number of vehicles.
        vehicle_capacities: The capacity of each vehicle.
        max_distance: The maximum distance a vehicle can travel.
        demands: The demands of each location.
        time_windows: The time windows for each location. Optional.
        backhauls: The pickup quantity for backhaul at each location.
    """

    depot: int
    distance_matrix: list[list[int]]
    duration_matrix: list[list[int]]
    num_vehicles: int
    vehicle_capacities: list[int]
    max_distance: int
    demands: list[int]
    time_windows: Optional[list[list[int]]]
    backhauls: Optional[list[int]]

    @property
    def num_locations(self) -> int:
        return len(self.distance_matrix)


def instance2data(instance: TensorDict) -> ORToolsData:
    """
    Converts an AnyVRP instance to an ORToolsData instance.
    """
    # TODO: Do not use PyVRP's data converter.
    data = pyvrp.instance2data(instance, ORTOOLS_SCALING_FACTOR)

    capacities = [
        veh_type.capacity
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    max_distance = data.vehicle_type(0).max_distance

    demands = [0] + [client.delivery for client in data.clients()]
    backhauls = [0] + [client.pickup for client in data.clients()]
    service = [0] + [client.service_duration for client in data.clients()]

    tws = [[data.location(0).tw_early, data.location(0).tw_late]]
    tws += [[client.tw_early, client.tw_late] for client in data.clients()]

    # Set data to None if instance does not contain explicit values.
    default_tw = [0, np.iinfo(np.int64).max]
    if all(tw == default_tw for tw in tws):
        tws = None  # type: ignore

    if all(val == 0 for val in backhauls):
        backhauls = None  # type: ignore

    distances = data.distance_matrix().copy()
    durations = np.array(distances) + np.array(service)[:, np.newaxis]

    if backhauls is not None:
        # Serve linehauls before backhauls.
        linehaul = np.flatnonzero(np.array(demands) > 0)
        backhaul = np.flatnonzero(np.array(backhauls) > 0)
        distances[np.ix_(backhaul, linehaul)] = max_distance

    return ORToolsData(
        depot=0,
        distance_matrix=distances.tolist(),
        duration_matrix=durations.tolist(),
        num_vehicles=data.num_vehicles,
        vehicle_capacities=capacities,
        demands=demands,
        time_windows=tws,
        max_distance=max_distance,
        backhauls=backhauls,
    )


def _solve(data: ORToolsData, max_runtime: float, log: bool = False):
    """
    Solves an instance with OR-Tools.

    Args:
        data: The instance data.
        max_runtime: The maximum runtime in seconds.
        log: Whether to log the search.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    # Manager for converting between nodes (location indices) and index
    # (internal CP variable indices).
    manager = pywrapcp.RoutingIndexManager(
        data.num_locations, data.num_vehicles, data.depot
    )
    routing = pywrapcp.RoutingModel(manager)

    # Set arc costs equal to distances.
    distance_transit_idx = routing.RegisterTransitMatrix(data.distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    # Max distance constraint.
    routing.AddDimension(
        distance_transit_idx,
        0,  # null distance slack
        data.max_distance,  # maximum distance per vehicle
        True,  # start cumul at zero
        "Distance",
    )

    # Vehicle capacity constraint.
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitVector(data.demands),
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Demand",
    )

    # Backhauls: this assumes that VRPB is implemented by forbidding arcs
    # that go from backhauls to linehauls.
    if data.backhauls is not None:
        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitVector(data.backhauls),
            0,  # null capacity slack
            data.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Backhaul",
        )

    # Time window constraints.
    if data.time_windows is not None:
        depot_tw_early = data.time_windows[data.depot][0]
        depot_tw_late = data.time_windows[data.depot][1]

        # The depot's late time window is a valid upper bound for the waiting
        # time and maximum duration per vehicle.
        routing.AddDimension(
            routing.RegisterTransitMatrix(data.duration_matrix),
            depot_tw_late,  # waiting time upper bound
            depot_tw_late,  # maximum duration per vehicle
            False,  # don't force start cumul to zero
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for node, (tw_early, tw_late) in enumerate(data.time_windows):
            if node == data.depot:  # skip depot
                continue

            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(tw_early, tw_late)

        # Add time window constraints for each vehicle start node.
        for node in range(data.num_vehicles):
            start = routing.Start(node)
            time_dim.CumulVar(start).SetRange(depot_tw_early, depot_tw_late)

        for node in range(data.num_vehicles):
            cumul_start = time_dim.CumulVar(routing.Start(node))
            routing.AddVariableMinimizedByFinalizer(cumul_start)

            cumul_end = time_dim.CumulVar(routing.End(node))
            routing.AddVariableMinimizedByFinalizer(cumul_end)

    # Setup search parameters.
    params = pywrapcp.DefaultRoutingSearchParameters()

    gls = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.local_search_metaheuristic = gls

    params.time_limit.FromSeconds(int(max_runtime))  # only accepts int
    params.log_search = log

    solution = routing.SolveWithParameters(params)
    action = solution2action(data, manager, routing, solution)
    objective = solution.ObjectiveValue()

    return action, objective


def solution2action(data, manager, routing, solution) -> list[list[int]]:
    """
    Converts an OR-Tools solution to routes.
    """
    routes = []
    distance = 0  # for debugging

    for vehicle_idx in range(data.num_vehicles):
        index = routing.Start(vehicle_idx)
        route = []
        route_cost = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx)

        if clients := route[1:]:  # ignore depot
            routes.append(clients)
            distance += route_cost

    return [visit for route in routes for visit in route + [0]]
