import math

from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.ops import get_distance_matrix
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class FLPGenerator(Generator):
    """Data generator for the Facility Location Problem (FLP).

    Args:
        num_loc: number of locations in the FLP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations
            orig_distances [batch_size, num_loc, num_loc]: original distances between locations
            distances [batch_size, num_loc]: the current minimum distance rom each location to the chosen locations
            chosen [batch_size, num_loc]: indicators of chosen locations
            to_choose [batch_size, 1]: number of locations to choose in the FLP
    """

    def __init__(
        self,
        num_loc: int = 100,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        to_choose: int = 10,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.to_choose = to_choose

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
        distances = get_distance_matrix(locs)
        max_dist = math.sqrt(2) * (self.max_loc - self.min_loc)

        return TensorDict(
            {
                "locs": locs,
                "orig_distances": distances,
                "distances": torch.full(
                    (*batch_size, self.num_loc), max_dist, dtype=torch.float
                ),
                "chosen": torch.zeros(*batch_size, self.num_loc, dtype=torch.bool),
                "to_choose": torch.ones(*batch_size, dtype=torch.long) * self.to_choose,
            },
            batch_size=batch_size,
        )
