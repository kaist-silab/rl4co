from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MTSPGenerator(Generator):
    """Data generator for the Multiple Travelling Salesman Problem (mTSP).

    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        min_num_agents: minimum number of agents (vehicles), include
        max_num_agents: maximum number of agents (vehicles), include

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            num_agents [batch_size]: number of agents (vehicles)
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        min_num_agents: int = 5,
        max_num_agents: int = 5,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample the number of agents
        num_agents = torch.randint(
            self.min_num_agents,
            self.max_num_agents + 1,
            size=(*batch_size,),
        )

        return TensorDict(
            {
                "locs": locs,
                "num_agents": num_agents,
            },
            batch_size=batch_size,
        )
