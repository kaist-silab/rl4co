from typing import Callable

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class PCTSPGenerator(Generator):
    """Data generator for the Prize-collecting Traveling Salesman Problem (PCTSP).

    Args:
        num_loc: number of locations (customers) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_demand: minimum value for the demand of each customer
        max_demand: maximum value for the demand of each customer
        demand_distribution: distribution for the demand of each customer
        capacity: capacity of the vehicle

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each city
            depot [batch_size, 2]: location of the depot
            demand [batch_size, num_loc]: demand of each customer
            capacity [batch_size, 1]: capacity of the vehicle
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = None,
        penalty_factor: float = 3.0,
        prize_required: float = 1.0,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.penalty_fctor = penalty_factor
        self.prize_required = prize_required

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
        self.deterministic_prize_sampler = get_sampler(
            "deterministric_prize", "uniform", 0.0, 4.0 / self.num_loc, **kwargs
        )
        self.stochastic_prize_sampler = get_sampler(
            "stochastic_prize", "uniform", 0.0, 2.0, **kwargs
        )

        # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
        # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
        # of the nodes by half of the tour length (which is very rough but similar to op)
        # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
        # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
        # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
        # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
        self.max_penalty = kwargs.get("max_penalty", None)
        if self.max_penalty is None:  # If not provided, use the default max penalty
            self.max_penalty = MAX_LENGTHS.get(num_loc, None)
        if (
            self.max_penalty is None
        ):  # If not in the table keys, find the closest number of nodes as the key
            closest_num_loc = min(MAX_LENGTHS.keys(), key=lambda x: abs(x - num_loc))
            self.max_penalty = MAX_LENGTHS[closest_num_loc]
            log.warning(
                f"The max penalty for {num_loc} locations is not defined. Using the closest max penalty: {self.max_penalty}\
                    with {closest_num_loc} locations."
            )

        # Adjust as in Kool et al. (2019)
        self.max_penalty *= penalty_factor / self.num_loc
        self.penalty_sampler = get_sampler(
            "penalty", "uniform", 0.0, self.max_penalty, **kwargs
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

        # Sample penalty
        penalty = self.penalty_sampler.sample((*batch_size, self.num_loc))

        # Take uniform prizes
        # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
        # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
        # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
        deterministic_prize = self.deterministic_prize_sampler.sample(
            (*batch_size, self.num_loc)
        )

        # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
        # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
        # stochastic prize is only revealed once the node is visited
        # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
        stochastic_prize = (
            self.stochastic_prize_sampler.sample((*batch_size, self.num_loc))
            * deterministic_prize
        )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "penalty": penalty,
                "deterministic_prize": deterministic_prize,
                "stochastic_prize": stochastic_prize,
            },
            batch_size=batch_size,
        )
