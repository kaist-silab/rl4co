from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from ..cvrp.env import CVRPEnv
from ..cvrp.generator import CVRPGenerator

log = get_pylogger(__name__)


class SDVRPEnv(CVRPEnv):
    """Split Delivery Vehicle Routing Problem (SDVRP) environment.
    SDVRP is a generalization of CVRP, where nodes can be visited multiple times and a fraction of the demand can be met.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is the -infinite unless the agent visits all the customers.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - location of the depot.
        - locations and demand/remaining demand of each customer
        - current location of the vehicle.
        - the remaining capacity of the vehicle.

    Constraints:
        - the tour starts and ends at the depot.
        - each customer can be visited multiple times.
        - the vehicle cannot visit customers exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has finished all customers demand and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "sdvrp"

    def __init__(
        self,
        generator: CVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(generator, generator_params, **kwargs)

    def _step(self, td: TensorDict) -> TensorDict:
        # Update the state
        current_node = td["action"][:, None]  # Add dimension for step

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand_with_depot"], current_node, dim=-1, squeeze=False
        )[..., :1]
        delivered_demand = torch.min(
            selected_demand, td["vehicle_capacity"] - td["used_capacity"]
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + delivered_demand) * (
            current_node != 0
        ).float()

        # Update demand
        demand_with_depot = td["demand_with_depot"].scatter_add(
            -1, current_node, -delivered_demand
        )

        # Get done
        done = ~(demand_with_depot > 0).any(-1)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        # Update state
        td.update(
            {
                "demand_with_depot": demand_with_depot,
                "current_node": current_node,
                "used_capacity": used_capacity,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Create reset TensorDict
        reset_td = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "demand_with_depot": torch.cat(
                    (torch.zeros_like(td["demand"][..., 0:1]), td["demand"]), -1
                ),
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
            },
            batch_size=batch_size,
        )
        reset_td.set("action_mask", self.get_action_mask(reset_td))
        return reset_td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        mask_loc = (td["demand_with_depot"][..., 1:] == 0) | (
            td["used_capacity"] >= td["vehicle_capacity"]
        )
        mask_depot = (td["current_node"] == 0).squeeze(-1) & (
            (mask_loc == 0).int().sum(-1) > 0
        )
        return ~torch.cat((mask_depot[..., None], mask_loc), -1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that the solution is valid (all demand is satisfied)"""

        batch_size, graph_size = td["demand"].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat((-td["vehicle_capacity"], td["demand"]), 1)

        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(td["demand"][..., 0])
        a_prev = None
        for a in actions.transpose(0, 1):
            assert (
                a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all()
            ), "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], td["vehicle_capacity"].squeeze(-1) - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

    def _make_spec(self, generator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=Bounded(
                low=generator.min_demand,
                high=generator.max_demand,
                shape=(generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
            ),
            demand_with_depot=Bounded(
                low=generator.min_demand,
                high=generator.max_demand,
                shape=(generator.num_loc + 1, 1),
                dtype=torch.float32,
            ),
            used_capacity=Bounded(
                low=0,
                high=generator.vehicle_capacity,
                shape=(1,),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)
