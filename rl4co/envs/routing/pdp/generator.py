from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PDPGenerator(Generator):
    """Data generator for the Pickup and Delivery Problem (PDP).
    Args:
        num_loc: number of locations (customers) in the PDP, without the depot. (e.g. 10 means 10 locs + 1 depot)
            - 1 depot
            - `num_loc` / 2 pickup locations
            - `num_loc` / 2 delivery locations
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        init_sol_type: str = "random",
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type

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

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
            },
            batch_size=batch_size,
        )

    # for improvement MDP only (to be refactored by a combination of rollout and the random policy)
    def _get_initial_solutions(self, coordinates):
        order_size = self.num_loc // 2
        batch_size = coordinates.size(0)

        if self.init_sol_type == "random":
            candidates = torch.ones(batch_size, self.num_loc + 1).bool()
            candidates[:, order_size + 1 :] = 0
            rec = torch.zeros(batch_size, self.num_loc + 1).long()
            selected_node = torch.zeros(batch_size, 1).long()
            candidates.scatter_(1, selected_node, 0)

            for i in range(self.num_loc):
                dists = torch.ones(batch_size, self.num_loc + 1)
                dists.scatter_(1, selected_node, -1e20)
                dists[~candidates] = -1e20
                dists = torch.softmax(dists, -1)
                next_selected_node = dists.multinomial(1).view(-1, 1)

                add_index = (next_selected_node <= order_size).view(-1)
                pairing = (
                    next_selected_node[next_selected_node <= order_size].view(-1, 1)
                    + order_size
                )
                candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)

                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

        elif self.init_sol_type == "greedy":
            candidates = torch.ones(batch_size, self.num_loc + 1).bool()
            candidates[:, order_size + 1 :] = 0
            rec = torch.zeros(batch_size, self.num_loc + 1).long()
            selected_node = torch.zeros(batch_size, 1).long()
            candidates.scatter_(1, selected_node, 0)

            for i in range(self.num_loc):
                d1 = coordinates.cpu().gather(
                    1, selected_node.unsqueeze(-1).expand(batch_size, self.num_loc + 1, 2)
                )
                d2 = coordinates.cpu()

                dists = (d1 - d2).norm(p=2, dim=2)
                dists.scatter_(1, selected_node, 1e6)
                dists[~candidates] = 1e6
                next_selected_node = dists.min(-1)[1].view(-1, 1)

                add_index = (next_selected_node <= order_size).view(-1)
                pairing = (
                    next_selected_node[next_selected_node <= order_size].view(-1, 1)
                    + order_size
                )
                candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)

                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

        else:
            raise NotImplementedError()

        return rec.expand(batch_size, self.num_loc + 1).clone()
