import os
import platform
from ctypes import (
    Structure, CDLL, POINTER, c_int, c_double, c_char, sizeof, cast, byref
)
from dataclasses import dataclass
from typing import List

import concurrent.futures
import numpy as np
import sys
import random
import time
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import get_distance_matrix
from rl4co.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


def get_lib_filename(hgs_dir: str) -> str:
    path = os.path.join(hgs_dir, "build", "libhgscvrp.so")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Shared library file `{path}` not found")
    return path


# Check if HGS-CVRP is installed
hgs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "HGS-CVRP")
try:
    HGS_LIBRARY_FILEPATH = get_lib_filename(hgs_dir)
except FileNotFoundError:
    log.info("HGS-CVRP is not installed. Running the build script...")
    os.popen(f"sh {hgs_dir}/build.sh").read()
    HGS_LIBRARY_FILEPATH = get_lib_filename(hgs_dir)
    log.info("HGS-CVRP is installed successfully.")


def local_search(td: TensorDict, actions: torch.Tensor, max_iterations: int = 1000) -> torch.Tensor:
    """
    Improve the solution using local search for CVRP, based on PyVRP.

    Args:
        td: TensorDict, td from env with shape [batch_size,]
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
        max_trials: int, maximum number of trials for local search
        allow_infeasible_solution: bool, whether to allow infeasible solutions
        seed: int, random seed for local search
    Returns:
        torch.Tensor, Improved tour indices with shape [batch_size, max_seq_len]
    """

    # Convert tensors to numpy arrays
    # Note: to avoid the overhead of device transfer, we recommend to pass the tensors in cpu
    actions_np = actions.detach().cpu().numpy()  # [batch_size, max_seq_len]
    actions_np = np.pad(actions_np, ((0, 0), (1, 1)), mode="constant")
    positions_np = td["locs"].detach().cpu().numpy()  # [batch_size, num_loc + 1, 2]
    demands_np = td["demand"].detach().cpu().numpy()  # [batch_size, num_loc]
    demands_np = np.pad(demands_np, ((0, 0), (1, 0)), mode="constant")  # Add depot demand
    distances = td.get("distances", None)  # [batch_size, num_loc + 1, num_loc + 1]
    if distances is None:
        distances_np = get_distance_matrix(td["locs"]).numpy()
    else:
        distances_np = distances.detach().cpu().numpy()

    subroutes_all: List[List[List[int]]] = [get_subroutes(path) for path in actions_np]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(subroutes_all)):
            future = executor.submit(
                swapstar,
                demands_np[i],
                distances_np[i],
                positions_np[i],
                subroutes_all[i],
                count=max_iterations,
            )
            futures.append((i, future))

    worst_seq_len = positions_np.shape[1] * 2
    new_actions = np.zeros((actions_np.shape[0], worst_seq_len), dtype=np.int64)

    for i, future in futures:
        new_actions[i] = merge_subroutes(future.result(), worst_seq_len)

    # Remove heading and tailing zeros
    max_pos = np.max(np.where(new_actions != 0)[1])
    new_actions = new_actions[:, 1: max_pos + 1]
    new_actions = torch.from_numpy(new_actions).to(td.device)

    # Check the validity of the solution and use the original solution if the new solution is invalid
    isvalid = check_validity(td, new_actions)
    if not isvalid.all():    
        new_actions[~isvalid] = 0
        orig_valid_actions = actions[~isvalid]
        # pad if needed
        orig_max_pos = torch.max(torch.where(orig_valid_actions != 0)[1]) + 1
        if orig_max_pos > max_pos:
            new_actions = torch.nn.functional.pad(
                new_actions, (0, orig_max_pos - max_pos, 0, 0), mode="constant", value=0  # type: ignore
            )
        new_actions[~isvalid, :orig_max_pos] = orig_valid_actions[:, :orig_max_pos]
    return new_actions


def get_subroutes(path, end_with_zero = True) -> List[List[int]]:
    x = np.where(path == 0)[0]
    subroutes = []
    for i, j in zip(x, x[1:]):
        if j - i > 1:
            if end_with_zero:
                j = j + 1
            subroutes.append(path[i: j])
    return subroutes


def merge_subroutes(subroutes, length):
    route = np.zeros(length, dtype=np.int64)
    i = 0
    for r in subroutes:
        if len(r) > 2:
            r = r[:-1]  # remove the last zero
            route[i: i + len(r)] = r
            i += len(r)
    return route


def check_validity(td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
    """
    Check the validity of the solution for CVRP and return a boolean tensor.
    Modified from CVRPEnv.check_solution_validity in rl4co/envs/routing/cvrp/env.py
    """
    # Check if tour is valid, i.e. contain 0 to n-1
    batch_size, graph_size = td["demand"].size()
    sorted_pi = actions.data.sort(1)[0]

    # Sorting it should give all zeros at front and then 1...n
    assert (
        torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
        .view(1, -1)
        .expand(batch_size, graph_size)
        == sorted_pi[:, -graph_size:]
    ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

    # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
    demand_with_depot = torch.cat((-td["vehicle_capacity"], td["demand"]), 1)
    d = demand_with_depot.gather(1, actions)

    used_cap = torch.zeros_like(td["demand"][:, 0])
    valid = torch.ones(batch_size, dtype=torch.bool)
    for i in range(actions.size(1)):
        used_cap += d[
            :, i
        ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
        # Cannot use less than 0
        used_cap[used_cap < 0] = 0
        valid &= (
            used_cap <= td["vehicle_capacity"][:, 0] + 1e-5
        )
    return valid


########################## HGS-CVRP python wrapper ###########################
# Adapted from https://github.com/chkwon/PyHygese/blob/master/hygese/hygese.py


c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)
C_INT_MAX = 2 ** (sizeof(c_int) * 8 - 1) - 1
C_DBL_MAX = sys.float_info.max


def write_routes(routes: List[np.ndarray], filepath: str):
    with open(filepath, "w") as f:
        for i, r in enumerate(routes):
            f.write(f"Route #{i + 1}: "+' '.join([str(x) for x in r if x > 0])+"\n")
    return


def read_routes(filepath) -> List[np.ndarray]:
    routes = []
    with open(filepath, "r") as f:
        while 1:
            line = f.readline().strip()
            if line.startswith("Route"):
                routes.append(np.array([0, *map(int, line.split(":")[1].split()), 0]))
            else:
                break
    return routes


# Must match with AlgorithmParameters.h in HGS-CVRP: https://github.com/vidalt/HGS-CVRP
class CAlgorithmParameters(Structure):
    _fields_ = [
        ("nbGranular", c_int),
        ("mu", c_int),
        ("lambda", c_int),
        ("nbElite", c_int),
        ("nbClose", c_int),
        ("targetFeasible", c_double),
        ("seed", c_int),
        ("nbIter", c_int),
        ("timeLimit", c_double),
        ("useSwapStar", c_int),
    ]


@dataclass
class AlgorithmParameters:
    nbGranular: int = 20
    mu: int = 25
    lambda_: int = 40
    nbElite: int = 4
    nbClose: int = 5
    targetFeasible: float = 0.2
    seed: int = 0
    nbIter: int = 20000
    timeLimit: float = 0.0
    useSwapStar: bool = True

    @property
    def ctypes(self) -> CAlgorithmParameters:
        return CAlgorithmParameters(
            self.nbGranular,
            self.mu,
            self.lambda_,
            self.nbElite,
            self.nbClose,
            self.targetFeasible,
            self.seed,
            self.nbIter,
            self.timeLimit,
            int(self.useSwapStar),
        )


class _SolutionRoute(Structure):
    _fields_ = [("length", c_int), ("path", c_int_p)]


class _Solution(Structure):
    _fields_ = [
        ("cost", c_double),
        ("time", c_double),
        ("n_routes", c_int),
        ("routes", POINTER(_SolutionRoute)),
    ]


class RoutingSolution:
    def __init__(self, sol_ptr):
        if not sol_ptr:
            raise TypeError("The solution pointer is null.")

        self.cost = sol_ptr[0].cost
        self.time = sol_ptr[0].time
        self.n_routes = sol_ptr[0].n_routes
        self.routes = []
        for i in range(self.n_routes):
            r = sol_ptr[0].routes[i]
            path = r.path[0 : r.length]
            self.routes.append(path)


class Solver:
    def __init__(self, parameters=AlgorithmParameters(), verbose=False):
        if platform.system() == "Windows":
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH, winmode=0)
        else:
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH)

        self.algorithm_parameters = parameters
        self.verbose = verbose

        # solve_cvrp
        self._c_api_solve_cvrp = hgs_library.solve_cvrp
        self._c_api_solve_cvrp.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
        ]
        self._c_api_solve_cvrp.restype = POINTER(_Solution)

        # solve_cvrp_dist_mtx
        self._c_api_local_search = hgs_library.local_search
        self._c_api_local_search.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
            c_int,
            c_int,
        ]
        self._c_api_local_search.restype = c_int

        # delete_solution
        self._c_api_delete_sol = hgs_library.delete_solution
        self._c_api_delete_sol.restype = None
        self._c_api_delete_sol.argtypes = [POINTER(_Solution)]
    
    def local_search(self, data, routes: List[np.ndarray], count: int = 1) -> List[np.ndarray]:
        # required data
        demand = np.asarray(data["demands"])
        vehicle_capacity = data["vehicle_capacity"]
        n_nodes = len(demand)

        # optional depot
        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        # optional num_vehicles
        maximum_number_of_vehicles = data.get("num_vehicles", C_INT_MAX)

        # optional service_times
        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes)
        else:
            service_times = np.asarray(service_times)

        # optional duration_limit
        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = C_DBL_MAX
        else:
            is_duration_constraint = True

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if x_coords is None or y_coords is None:
            assert dist_mtx is not None
            x_coords = np.zeros(n_nodes)
            y_coords = np.zeros(n_nodes)
        else:
            x_coords = np.asarray(x_coords)
            y_coords = np.asarray(y_coords)

        assert len(x_coords) == len(y_coords) == len(service_times) == len(demand)
        assert (x_coords >= 0.0).all()
        assert (y_coords >= 0.0).all()
        assert (service_times >= 0.0).all()
        assert (demand >= 0.0).all()

        dist_mtx = np.asarray(dist_mtx)
        assert dist_mtx.shape[0] == dist_mtx.shape[1]
        assert (dist_mtx >= 0.0).all()

        callid = (time.time_ns()*10000+random.randint(0,10000))%C_INT_MAX

        tmppath = "/tmp/route-{}".format(callid)
        resultpath = "/tmp/swapstar-result-{}".format(callid)
        write_routes(routes, tmppath)
        try:
            self._local_search(
                x_coords,
                y_coords,
                dist_mtx,
                service_times,
                demand,
                vehicle_capacity,
                duration_limit,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters,
                self.verbose,
                callid,
                count,
            )
            result = read_routes(resultpath)
        except Exception as e:
            print(e)
            result = routes
        else:
            os.remove(resultpath)
        finally:
            os.remove(tmppath)
        
        return result

    def _local_search(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        dist_mtx: np.ndarray,
        service_times: np.ndarray,
        demand: np.ndarray,
        vehicle_capacity: int,
        duration_limit: float,
        is_duration_constraint: bool,
        maximum_number_of_vehicles: int,
        algorithm_parameters: AlgorithmParameters,
        verbose: bool,
        callid: int,
        count:int,
    ):
        n_nodes = x_coords.size

        x_ct = x_coords.astype(c_double).ctypes
        y_ct = y_coords.astype(c_double).ctypes
        s_ct = service_times.astype(c_double).ctypes
        d_ct = demand.astype(c_double).ctypes

        m_ct = dist_mtx.reshape(n_nodes * n_nodes).astype(c_double).ctypes
        ap_ct = algorithm_parameters.ctypes


        # struct Solution *solve_cvrp_dist_mtx(
        # 	int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
        # 	double vehicleCapacity, double durationLimit, char isDurationConstraint,
        # 	int max_nbVeh, const struct AlgorithmParameters *ap, char verbose);
        sol_p = self._c_api_local_search(
            n_nodes,
            cast(x_ct, c_double_p),  # type: ignore
            cast(y_ct, c_double_p),  # type: ignore
            cast(m_ct, c_double_p),  # type: ignore
            cast(s_ct, c_double_p),  # type: ignore
            cast(d_ct, c_double_p),  # type: ignore
            vehicle_capacity,
            duration_limit,
            is_duration_constraint,
            maximum_number_of_vehicles,
            byref(ap_ct),
            verbose,
            callid,
            count,
        )

        result = sol_p
        return result


def swapstar(demands, matrix, positions, routes: List[np.ndarray], count=1):
    ap = AlgorithmParameters()
    hgs_solver = Solver(parameters=ap, verbose=False)

    data = dict()
    x = positions[:, 0]
    y = positions[:, 1]
    data['x_coordinates'] = x
    data['y_coordinates'] = y

    data['depot'] = 0
    data['demands'] = demands * 1000
    data["num_vehicles"] = len(routes)
    data['vehicle_capacity'] = 1000.001  # to avoid floating-point error

    # Solve with calculated distances
    data['distance_matrix'] = matrix
    result = hgs_solver.local_search(data, routes, count)
    return result
