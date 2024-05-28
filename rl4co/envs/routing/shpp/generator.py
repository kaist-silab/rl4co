from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)


class SHPPGenerator(Generator):
    """Data generator for the Shortest Hamiltonian Path Problem (SHPP).
    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler("loc", loc_distribution, min_loc, max_loc, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        
        return TensorDict(
            {
                "locs": locs,
            },
            batch_size=batch_size,
        )
