from typing import Optional

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from .generator import SHPPGenerator
from .render import render

log = get_pylogger(__name__)


class SHPPEnv(RL4COEnvBase):
    """
    Shortest Hamiltonian Path Problem (SHPP)
    SHPP is referred to the open-loop Traveling Salesman Problem (TSP) in the literature.
    The goal of the SHPP is to find the shortest Hamiltonian path in a given graph with
    given fixed starting/terminating nodes (they can be different nodes). A Hamiltonian
    path visits all other nodes exactly once. At each step, the agent chooses a city to visit.
    The reward is 0 unless the agent visits all the cities. In that case, the reward is
    (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observation:
        - locations of each customer
        - starting node and terminating node
        - the current location of the vehicle

    Constraints:
        - the first node is the starting node
        - the last node is the terminating node
        - each node is visited exactly once

    Finish condition:
        - the agent has visited all the customers and reached the terminating node

    Reward:
        - (minus) the length of the path

    Args:
        generator: SHPPGenerator instance as the generator
        generator_params: parameters for the generator
    """

    name = "shpp"

    def __init__(
        self,
        generator: SHPPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SHPPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # If all other nodes are visited, the terminating node will be available
        action_mask = available.clone()
        action_mask[..., -1] = ~available[..., :-1].any(dim=-1)

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "available": available,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Note: the first node is the starting node; the last node is the terminating node"""
        device = td.device
        locs = td["locs"]

        # We do not enforce loading from self for flexibility
        num_loc = locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        last_node = torch.full(
            (batch_size), num_loc - 1, dtype=torch.int64, device=device
        )
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        action_mask = torch.zeros((*batch_size, num_loc), dtype=torch.bool, device=device)
        action_mask[..., 0] = 1  # Only the start point is availabe at the beginning
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": locs,
                "first_node": current_node,
                "last_node": last_node,
                "current_node": current_node,
                "i": i,
                "available": available,
                "action_mask": action_mask,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _get_reward(self, td, actions) -> TensorDict:
        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

    def _make_spec(self, generator):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
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
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
