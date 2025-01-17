from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SVRPGenerator(Generator):
    """Data generator for the Skill Vehicle Routing Problem (SVRP).
    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations        min_skill: minimum value for the technic skill
        max_skill: maximum value for the technic skill
        skill_distribution: distribution for the technic skill
        tech_costs: list of the technic costs

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
            techs [batch_size, num_loc]: technic requirements of each customer
            skills [batch_size, num_loc]: skills of the vehicles
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        min_skill: float = 1.0,
        max_skill: float = 10.0,
        tech_costs: list = [1, 2, 3],
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_skill = min_skill
        self.max_skill = max_skill
        self.num_tech = len(tech_costs)
        self.tech_costs = torch.tensor(tech_costs)

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
        """Generate data for the basic Skill-VRP. The data consists of the locations of the customers,
        the skill-levels of the technicians and the required skill-levels of the customers.
        The data is generated randomly within the given bounds."""
        # Sample locations: depot and customers
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        locs_with_depot = torch.cat((depot[:, None, :], locs), dim=1)

        # Initialize technicians and sort ascendingly
        techs, _ = torch.sort(
            torch.FloatTensor(*batch_size, self.num_tech, 1).uniform_(
                self.min_skill, self.max_skill
            ),
            dim=-2,
        )

        # Initialize the skills
        skills = torch.FloatTensor(*batch_size, self.num_loc, 1).uniform_(0, 1)
        # scale skills
        skills = torch.max(techs, dim=1, keepdim=True).values * skills
        td = TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "techs": techs,
                "skills": skills,
            },
            batch_size=batch_size,
        )
        return td
