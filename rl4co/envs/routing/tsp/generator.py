from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TSPGenerator(Generator):
    """Data generator for the Travelling Salesman Problem (TSP).

    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        init_sol_type: the method type used for generating initial solutions (random or greedy)
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
        init_sol_type: str = "random",
        loc_distribution: int | float | str | type | Callable = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type

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

        return TensorDict(
            {
                "locs": locs,
            },
            batch_size=batch_size,
        )

    # for improvement MDP only (to be refactored by a combination of rollout and the random policy)
    def _get_initial_solutions(self, coordinates):
        batch_size = coordinates.size(0)

        if self.init_sol_type == "random":
            set = torch.rand(batch_size, self.num_loc).argsort().long()
            rec = torch.zeros(batch_size, self.num_loc).long()
            index = torch.zeros(batch_size, 1).long()

            for i in range(self.num_loc - 1):
                rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))

            rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))

        elif self.init_sol_type == "greedy":
            candidates = torch.ones(batch_size, self.num_loc).bool()
            rec = torch.zeros(batch_size, self.num_loc).long()
            selected_node = torch.zeros(batch_size, 1).long()
            candidates.scatter_(1, selected_node, 0)

            for i in range(self.num_loc - 1):
                d1 = coordinates.cpu().gather(
                    1, selected_node.unsqueeze(-1).expand(batch_size, self.num_loc, 2)
                )
                d2 = coordinates.cpu()

                dists = (d1 - d2).norm(p=2, dim=2)
                dists[~candidates] = 1e5

                next_selected_node = dists.min(-1)[1].view(-1, 1)
                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

        else:
            raise NotImplementedError()

        return rec.expand(batch_size, self.num_loc).clone()
