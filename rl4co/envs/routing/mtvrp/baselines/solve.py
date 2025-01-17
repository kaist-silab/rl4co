from functools import partial
from multiprocessing import Pool

from tensordict.tensordict import TensorDict
from torch import Tensor

from .utils import process_instance


class NoSolver:
    def solve(self, *args, **kwargs):
        pass


try:
    import rl4co.envs.routing.mtvrp.baselines.pyvrp as pyvrp
except ImportError:
    pyvrp = NoSolver()
try:
    import rl4co.envs.routing.mtvrp.baselines.lkh as lkh
except ImportError:
    lkh = NoSolver()
try:
    import rl4co.envs.routing.mtvrp.baselines.ortools as ortools
except ImportError:
    ortools = NoSolver()


def solve(
    instances: TensorDict,
    max_runtime: float,
    num_procs: int = 1,
    solver: str = "pyvrp",
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instances with PyVRP.

    Args:
        instances: The AnyVRP instances to solve.
        max_runtime: The maximum runtime for the solver.
        num_procs: The number of processes to use.
        solver: The solver to use.

    Returns:
        A tuple containing the action and the cost, respectively.
    """

    instances = process_instance(instances)

    if solver == "pyvrp" and isinstance(pyvrp, NoSolver):
        raise ImportError(
            "PyVRP is not installed. Please install it using `pip install -e .[solvers]`."
        )
    if solver == "lkh" and isinstance(lkh, NoSolver):
        raise ImportError(
            "LKH is not installed. Please install it using `pip install -e .[solvers]`"
        )
    if solver == "ortools" and isinstance(ortools, NoSolver):
        raise ImportError(
            "OR-Tools is not installed. Please install it using `pip install -e .[solvers]`."
        )

    solvers = {"pyvrp": pyvrp.solve, "ortools": ortools.solve, "lkh": lkh.solve}
    if solver not in solvers:
        raise ValueError(f"Unknown baseline solver: {solver}")

    _solve = solvers[solver]
    func = partial(_solve, max_runtime=max_runtime, **kwargs)

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            results = pool.map(func, instances)
    else:
        results = [func(instance) for instance in instances]

    actions, costs = zip(*results)

    # Pad to ensure all actions have the same length.
    max_len = max(len(action) for action in actions)
    actions = [action + [0] * (max_len - len(action)) for action in actions]

    return Tensor(actions).long(), Tensor(costs)
