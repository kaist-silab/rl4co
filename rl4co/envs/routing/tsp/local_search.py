import concurrent.futures

import numpy as np
import numba as nb
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import get_distance_matrix
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def local_search(td: TensorDict, actions: torch.Tensor, max_iterations: int = 1000) -> torch.Tensor:
    """
    Improve the solution using local search, especially 2-opt for TSP.
    Implementation credits to: https://github.com/henry-yeh/DeepACO

    Args:
        td: TensorDict, td from env with shape [batch_size,]
        actions: torch.Tensor, Tour indices with shape [batch_size, num_loc]
        max_iterations: int, maximum number of iterations for 2-opt
    Returns:
        torch.Tensor, Improved tour indices with shape [batch_size, num_loc]
    """
    distances = td.get("distances", None)
    if distances is None:
        distances_np = get_distance_matrix(td["locs"]).numpy()
    else:
        distances_np = distances.detach().cpu().numpy()
    distances_np = distances_np + 1e9 * np.eye(distances_np.shape[1], dtype=np.float32)[None, :, :]

    actions_np = actions.detach().cpu().numpy().astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for dist, tour in zip(distances_np, actions_np):
            future = executor.submit(_two_opt_python, distmat=dist, tour=tour, max_iterations=max_iterations)
            futures.append(future)
        return torch.from_numpy(np.stack([f.result() for f in futures]).astype(np.int64)).to(actions.device)


@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j] + distmat[node_i, node_next]
                - distmat[node_prev, node_i] - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
    return tour
