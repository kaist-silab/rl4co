from typing import Optional

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import PCTSPGenerator
from .render import render

log = get_pylogger(__name__)


class PCTSPEnv(RL4COEnvBase):
    """Prize-collecting TSP (PCTSP) environment.
    The goal is to collect as much prize as possible while minimizing the total travel cost.
    The environment is stochastic, the prize is only revealed when the node is visited.

    Observations:
        - locations of the nodes
        - prize and penalty of each node
        - current location of the vehicle
        - current total prize
        - current total penalty
        - visited nodes
        - prize required to visit a node
        - the current step

    Constraints:
        - the tour starts and ends at the depot
        - the vehicle cannot visit nodes exceed the remaining prize

    Finish Condition:
        - the vehicle back to the depot

    Reward:
        - the sum of the saved penalties

    Args:
        generator: OPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "pctsp"
    _stochastic = False

    def __init__(
        self,
        generator: PCTSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = PCTSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # Get current coordinates, prize, and penalty
        cur_total_prize = td["cur_total_prize"] + gather_by_index(
            td["real_prize"], current_node
        )
        cur_total_penalty = td["cur_total_penalty"] + gather_by_index(
            td["penalty"], current_node
        )

        # Update visited
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # Done and reward
        done = (td["i"] > 0) & (current_node == 0)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        # Update state
        td.update(
            {
                "current_node": current_node,
                "cur_total_prize": cur_total_prize,
                "cur_total_penalty": cur_total_penalty,
                "visited": visited,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device

        locs = torch.cat([td["depot"][..., None, :], td["locs"]], dim=-2)
        expected_prize = td["deterministic_prize"]
        real_prize = (
            td["stochastic_prize"] if self.stochastic else td["deterministic_prize"]
        )
        penalty = td["penalty"]

        # Concatenate depots
        real_prize_with_depot = torch.cat(
            [torch.zeros_like(real_prize[..., :1]), real_prize], dim=-1
        )
        penalty_with_depot = F.pad(penalty, (1, 0), mode="constant", value=0)

        # Initialize the current node and  prize / penalty
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        cur_total_prize = torch.zeros(*batch_size, device=device)
        cur_total_penalty = penalty.sum(-1)  # Sum penalties (all when nothing is visited)

        # Init the action mask (all nodes are available)
        visited = torch.zeros(
            (*batch_size, self.generator.num_loc + 1), dtype=torch.bool, device=device
        )
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        prize_required = torch.full(
            (*batch_size,), self.generator.prize_required, device=device
        )

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "expected_prize": expected_prize,
                "real_prize": real_prize_with_depot,
                "penalty": penalty_with_depot,
                "cur_total_prize": cur_total_prize,
                "cur_total_penalty": cur_total_penalty,
                "visited": visited,
                "prize_required": prize_required,
                "i": i,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Cannot visit depot if not yet collected 1 total prize and there are unvisited nodes"""
        mask = td["visited"] | td["visited"][..., 0:1]
        mask[..., 0] = (td["cur_total_prize"] < 1.0) & (
            td["visited"][..., 1:].int().sum(-1) < td["visited"][..., 1:].size(-1)
        )
        return ~(mask > 0)  # Invert mask, since 1 means feasible action

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Reward is `saved penalties - (total length + penalty)`"""

        # In case all tours directly return to depot, prevent further problems
        if actions.size(-1) == 1:
            assert (actions == 0).all(), "If all length 1 tours, they should be zero"
            return torch.zeros(actions.size(0), dtype=torch.float, device=actions.device)

        # Gather locations in order of tour (add depot since we start and end there)
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],  # depot
                gather_by_index(td["locs"], actions),  # order locations
            ],
            dim=1,
        )
        length = get_tour_length(locs_ordered)

        # Reward is saved penalties - (total length + penalty)
        saved_penalty = td["penalty"].gather(1, actions)
        return saved_penalty.sum(-1) - (length + td["penalty"][..., 1:].sum(-1))

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that the solution is valid, i.e. contains all nodes once at most, and either prize constraint is met or all nodes are visited"""

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_actions = actions.data.sort(1)[0]

        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[..., 1:] == 0)
            | (sorted_actions[..., 1:] > sorted_actions[..., :-1])
        ).all(), "Duplicates"

        prize = td["real_prize"][..., 1:]  # Remove depot
        prize_with_depot = torch.cat((torch.zeros_like(prize[:, :1]), prize), 1)
        p = prize_with_depot.gather(1, actions)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (p.sum(-1) >= 1 - 1e-5)
            | (
                sorted_actions.size(-1) - (sorted_actions == 0).int().sum(-1)
                == (td["locs"].size(-2) - 1)
            )  # no depot
        ).all(), "Total prize does not satisfy min total prize"

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is True:
            log.warning(
                "Stochastic mode should not be used for PCTSP. Use SPCTSP instead."
            )

    def _make_spec(self, generator):
        """Make the locs and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            expected_prize=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.float32,
            ),
            real_prize=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.float32,
            ),
            penalty=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.float32,
            ),
            cur_total_prize=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            cur_total_penalty=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            visited=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            prize_required=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
