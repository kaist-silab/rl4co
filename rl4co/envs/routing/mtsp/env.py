from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import MTSPGenerator
from .render import render

log = get_pylogger(__name__)


class MTSPEnv(RL4COEnvBase):
    """Multiple Traveling Salesman Problem (mTSP) environment.
    At each step, an agent chooses to visit a city. A maximum of `num_agents` agents can be employed to visit the cities.

    Observations:
        - locations of the depot and each city.
        - number of agents.
        - the current agent index.
        - the current location of the vehicle.

    Constrains:
        - each agent's tour starts and ends at the depot.
        - each city must be visited exactly once.

    Finish condition:
        - all cities are visited and all agents back to the depot.

    Reward:
        There are two ways to calculate the cost (-reward):
        - `minmax`: (default) the cost is the maximum of the path lengths of all the agents.
        - `sum`: the cost is the sum of the path lengths of all the agents.

    Args:
        cost_type: type of cost to use, either `minmax` or `sum`
        generator: MTSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "mtsp"

    def __init__(
        self,
        cost_type: str = "minmax",
        generator: MTSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MTSPGenerator(**generator_params)
        self.generator = generator
        self.cost_type = cost_type
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # Initial variables
        is_first_action = batch_to_scalar(td["i"]) == 0
        current_node = td["action"]
        first_node = current_node if is_first_action else td["first_node"]

        # Get the locations of the current node and the previous node and the depot
        cur_loc = gather_by_index(td["locs"], current_node)
        prev_loc = gather_by_index(
            td["locs"], td["current_node"]
        )  # current_node is the previous node
        depot_loc = td["locs"][..., 0, :]

        # If current_node is the depot, then increment agent_idx
        cur_agent_idx = td["agent_idx"] + (current_node == 0).long()

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node[..., None].expand_as(td["action_mask"]), 0
        )
        # Available[..., 0] is the depot, which is always available unless:
        # - current_node is the depot
        # - agent_idx greater than num_agents -1
        available[..., 0] = torch.logical_and(
            current_node != 0, td["agent_idx"] < td["num_agents"] - 1
        )

        # We are done there are no unvisited locations except the depot
        done = torch.count_nonzero(available[..., 1:], dim=-1) == 0

        # If done is True, then we make the depot available again, so that it will be selected as the next node with prob 1
        available[..., 0] = torch.logical_or(done, available[..., 0])

        # Update the current length
        current_length = td["current_length"] + get_distance(cur_loc, prev_loc)

        # If done, we add the distance from the current_node to the depot as well
        current_length = torch.where(
            done, current_length + get_distance(cur_loc, depot_loc), current_length
        )

        # We update the max_subtour_length and reset the current_length
        max_subtour_length = torch.where(
            current_length > td["max_subtour_length"],
            current_length,
            td["max_subtour_length"],
        )

        # If current agent is different from previous agent, then we have a new subtour and reset the length
        current_length *= (cur_agent_idx == td["agent_idx"]).float()

        # The reward is the negative of the max_subtour_length (minmax objective)
        reward = -max_subtour_length

        td.update(
            {
                "max_subtour_length": max_subtour_length,
                "current_length": current_length,
                "agent_idx": cur_agent_idx,
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            }
        )

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = self.device
        locs = td["locs"]
        num_loc = locs.size(-2)

        agent_idx = torch.zeros((*batch_size, ), dtype=torch.int64, device=device)
        max_subtour_length = torch.zeros((*batch_size, ), dtype=torch.float32, device=device)
        current_length = torch.zeros((*batch_size, ), dtype=torch.float32, device=device)

        first_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size, ), dtype=torch.int64, device=device)
        action_mask = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        reward = torch.zeros((*batch_size, ), dtype=torch.float32, device=device)

        td_reset = TensorDict(
            {
                "locs": locs,
                "first_node": first_node,
                "current_node": first_node,
                "num_agents": td["num_agents"],
                "max_subtour_length": max_subtour_length,
                "current_length": current_length,
                "agent_idx": agent_idx,
                "i": i,
                "action_mask": action_mask,
                "reward": reward,
            },
            batch_size=batch_size,
        )
        return td_reset

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> torch.Tensor:
        # With minmax, get the maximum distance among subtours, calculated in the model
        if self.cost_type == "minmax":
            return td["reward"].squeeze(-1)

        # With distance, same as TSP
        elif self.cost_type == "sum":
            locs = td["locs"]
            locs_ordered = locs.gather(1, actions.unsqueeze(-1).expand_as(locs))
            return -get_tour_length(locs_ordered)

        else:
            raise ValueError(f"Cost type {self.cost_type} not supported")

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        log.warning("check_solution_validity not implemented for mTSP")

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)

    def _make_spec(self, generator: MTSPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            num_agents=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            agent_idx=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            max_subtour_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec()
        self.done_spec = UnboundedDiscreteTensorSpec(dtype=torch.bool)
