from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MDCPDPGenerator(Generator):
    """Data generator for the Multi Depot Capacitated Pickup and Delivery Problem (MDCPDP) environment.

    Args:
        num_loc: number of locations (customers)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates, default is 150 insted of 1.0, will be scaled
        loc_distribution: distribution for the location coordinates
        num_depot: number of depots, each depot has one vehicle
        depot_mode: mode for the depot, either single or multiple
        depod_distribution: distribution for the depot coordinates
        min_capacity: minimum value of the capacity
        max_capacity: maximum value of the capacity
        min_lateness_weight: minimum value of the lateness weight
        max_lateness_weight: maximum value of the lateness weight
        latebess_weight_distribution: distribution for the lateness weight

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, num_depot, 2]: locations of each depot
            capacity [batch_size, 1]: capacity of the vehicle
            lateness_weight [batch_size, 1]: weight of the lateness cost
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        num_depot: int = 5,
        depot_mode: str = "multiple",
        depot_distribution: int | float | str | type | Callable = Uniform,
        min_capacity: int = 1,
        max_capacity: int = 5,
        min_lateness_weight: float = 1.0,
        max_lateness_weight: float = 1.0,
        lateness_weight_distribution: int | float | str | type | Callable = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.depot_mode = depot_mode
        self.num_depot = num_depot
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.min_lateness_weight = min_lateness_weight
        self.max_lateness_weight = max_lateness_weight

        # Number of locations must be even
        if num_loc % 2 != 0:
            log.warn(
                "Number of locations must be even. Adding 1 to the number of locations."
            )
            self.num_loc += 1

        # Check depot mode validity
        assert depot_mode in ["single", "multiple"], f"Invalid depot mode: {depot_mode}"

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
            self.depot_sampler = get_sampler(
                "depot", depot_distribution, min_loc, max_loc, **kwargs
            )

        # Lateness weight distribution
        if kwargs.get("lateness_weight_sampler", None) is not None:
            self.lateness_weight_sampler = kwargs["lateness_weight_sampler"]
        else:
            self.lateness_weight_sampler = get_sampler(
                "lateness_weight",
                lateness_weight_distribution,
                min_lateness_weight,
                max_lateness_weight,
                **kwargs,
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample depot
        if self.depot_mode == "single":
            depot = self.depot_sampler.sample((*batch_size, 2))[:, None, :].repeat(
                1, self.num_depot, 1
            )
        else:
            depot = self.depot_sampler.sample((*batch_size, self.num_depot, 2))

        # Sample capacity
        capacity = torch.randint(
            self.min_capacity,
            self.max_capacity + 1,
            size=(*batch_size, 1),
        )

        # Sample lateness weight
        lateness_weight = self.lateness_weight_sampler.sample((*batch_size, 1))

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "capacity": capacity,
                "lateness_weight": lateness_weight,
            },
            batch_size=batch_size,
        )
