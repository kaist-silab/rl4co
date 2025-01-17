from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ATSPGenerator(Generator):
    """Data generator for the Asymmetric Travelling Salesman Problem (ATSP)
    Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
    We satifsy the triangle inequality (TMAT class) in a batch

    Args:
        num_loc: number of locations (customers) in the TSP
        min_dist: minimum value for the distance between nodes
        max_dist: maximum value for the distance between nodes
        dist_distribution: distribution for the distance between nodes
        tmat_class: whether to generate a class of distance matrix

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """

    def __init__(
        self,
        num_loc: int = 10,
        min_dist: float = 0.0,
        max_dist: float = 1.0,
        dist_distribution: int | float | str | type | Callable = Uniform,
        tmat_class: bool = True,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.tmat_class = tmat_class

        # Distance distribution
        if kwargs.get("dist_sampler", None) is not None:
            self.dist_sampler = kwargs["dist_sampler"]
        else:
            self.dist_sampler = get_sampler("dist", dist_distribution, 0.0, 1.0, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
            self.dist_sampler.sample((batch_size + [self.num_loc, self.num_loc]))
            * (self.max_dist - self.min_dist)
            + self.min_dist
        )
        dms[..., torch.arange(self.num_loc), torch.arange(self.num_loc)] = 0
        log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            for i in range(self.num_loc):
                dms = torch.minimum(dms, dms[..., :, [i]] + dms[..., [i], :])
        return TensorDict({"cost_matrix": dms}, batch_size=batch_size)
