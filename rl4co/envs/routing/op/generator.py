from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# From Kool et al. 2019
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class OPGenerator(Generator):
    """Data generator for the Orienteering Problem (OP).

    Args:
        num_loc: number of locations (customers) in the OP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_prize: minimum value for the prize of each customer
        max_prize: maximum value for the prize of each customer
        prize_distribution: distribution for the prize of each customer
        max_length: maximum length of the path

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
            prize [batch_size, num_loc]: prize of each customer
            max_length [batch_size, 1]: maximum length of the path for each customer
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        min_prize: float = 1.0,
        max_prize: float = 1.0,
        prize_distribution: int | float | type | Callable = Uniform,
        prize_type: str = "dist",
        max_length: float | torch.Tensor = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.prize_type = prize_type
        self.max_length = max_length

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

        # Prize distribution
        if kwargs.get("prize_sampler", None) is not None:
            self.prize_sampler = kwargs["prize_sampler"]
        elif (
            prize_distribution == "dist"
        ):  # If prize_distribution is 'dist', then the prize is the distance from the depot
            self.prize_sampler = None
        else:
            self.prize_sampler = get_sampler(
                "prize", prize_distribution, min_prize, max_prize, **kwargs
            )

        # Max length
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = MAX_LENGTHS.get(num_loc, None)
        if self.max_length is None:
            closest_num_loc = min(MAX_LENGTHS.keys(), key=lambda x: abs(x - num_loc))
            self.max_length = MAX_LENGTHS[closest_num_loc]
            log.warning(
                f"The max length for {num_loc} locations is not defined. Using the closest max length: {self.max_length}\
                    with {closest_num_loc} locations."
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

        locs_with_depot = torch.cat((depot.unsqueeze(1), locs), dim=1)

        # Methods taken from Fischetti et al. (1998) and Kool et al. (2019)
        if self.prize_type == "const":
            prize = torch.ones(*batch_size, self.num_loc, device=self.device)
        elif self.prize_type == "unif":
            prize = (
                1
                + torch.randint(
                    0, 100, (*batch_size, self.num_loc), device=self.device
                ).float()
            ) / 100
        elif self.prize_type == "dist":  # based on the distance to the depot
            prize = (locs_with_depot[..., 0:1, :] - locs_with_depot[..., 1:, :]).norm(
                p=2, dim=-1
            )
            prize = (
                1 + (prize / prize.max(dim=-1, keepdim=True)[0] * 99).int()
            ).float() / 100
        else:
            raise ValueError(f"Invalid prize_type: {self.prize_type}")

        # Support for heterogeneous max length if provided
        if not isinstance(self.max_length, torch.Tensor):
            max_length = torch.full((*batch_size,), self.max_length)
        else:
            max_length = self.max_length

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "prize": prize,
                "max_length": max_length,
            },
            batch_size=batch_size,
        )
