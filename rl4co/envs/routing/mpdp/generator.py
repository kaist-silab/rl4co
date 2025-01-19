from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MPDPGenerator(Generator):
    """Data generator for the Capacitated Vehicle Routing Problem (CVRP).
    Args:
        num_loc: number of locations
        min_loc: minimum location value
        max_loc: maximum location value
        loc_distribution: distribution for the locations
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_num_agents: minimum number of agents
        max_num_agents: maximum number of agents

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer and the depot
            depot [batch_size, 2]: location of the depot
            num_agents [batch_size]: number of agents
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        min_num_agents: int = 2,
        max_num_agents: int = 10,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents

        # Number of locations must be even
        if num_loc % 2 != 0:
            log.warn(
                "Number of locations must be even. Adding 1 to the number of locations."
            )
            self.num_loc += 1

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        # Sample the number of agents
        num_agents = torch.randint(
            self.min_num_agents,
            self.max_num_agents + 1,
            size=(*batch_size,),
        )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "num_agents": num_agents,
            },
            batch_size=batch_size,
        )
