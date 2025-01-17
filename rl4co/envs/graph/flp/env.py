from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .generator import FLPGenerator

log = get_pylogger(__name__)


class FLPEnv(RL4COEnvBase):
    """Facility Location Problem (FLP) environment
    At each step, the agent chooses a location. The reward is 0 unless enough number of locations are chosen.
    The reward is (-) the total distance of each location to its closest chosen location.

    Observations:
        - the locations
        - the number of locations to choose

    Constraints:
        - the given number of locations must be chosen

    Finish condition:
        - the given number of locations are chosen

    Reward:
        - (minus) the total distance of each location to its closest chosen location

    Args:
        generator: FLPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "flp"

    def __init__(
        self,
        generator: FLPGenerator = None,
        generator_params: dict = {},
        check_solution=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = FLPGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, 1]; the location to be chosen in each instance
        selected = td["action"]
        batch_size = selected.shape[0]

        # Update location selection status
        chosen = td["chosen"].clone()  # (batch_size, n_locations)
        n_points_ = chosen.shape[-1]

        chosen[torch.arange(batch_size).to(td.device), selected] = True

        # We are done if we choose enough locations
        done = td["i"] >= (td["to_choose"] - 1)

        # The reward is calculated outside via get_reward for efficiency, so we set it to zero here
        reward = torch.zeros_like(done)

        # Update distances
        orig_distances = td["orig_distances"]  # (batch_size, n_points, n_points)

        cur_min_dist = (
            gather_by_index(
                orig_distances, chosen.nonzero(as_tuple=True)[1].view(batch_size, -1)
            )
            .view(batch_size, -1, n_points_)
            .min(dim=1)
            .values
        )

        # We cannot choose the already-chosen locations
        action_mask = ~chosen

        td.update(
            {
                "distances": cur_min_dist,  # (batch_size, n_points)
                # states changed by actions
                "chosen": chosen,  # each entry is binary; 1 iff the corresponding facility is chosen
                "i": td["i"] + 1,  # the number of sets we have chosen
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        self.to(td.device)

        return TensorDict(
            {
                # given information
                "locs": td["locs"],  # (batch_size, n_points, dim_loc)
                "orig_distances": td[
                    "orig_distances"
                ],  # (batch_size, n_points, n_points)
                "distances": td["distances"],  # (batch_size, n_points, n_points)
                # states changed by actions
                "chosen": torch.zeros(
                    *td["locs"].shape[:-1], dtype=torch.bool, device=td.device
                ),  # each entry is binary; 1 iff the corresponding facility is chosen
                "to_choose": td["to_choose"],  # the number of sets to choose
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),  # the number of sets we have chosen
                "action_mask": torch.ones(
                    *td["locs"].shape[:-1], dtype=torch.bool, device=td.device
                ),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: FLPGenerator):
        # TODO: make spec
        pass

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # The reward is (minus) the total distance from each location to the closest chosen location
        chosen = td["chosen"]  # (batch_size, n_points)
        batch_size_ = td["chosen"].shape[0]
        n_points_ = td["chosen"].shape[-1]
        orig_distances = td["orig_distances"]
        cur_min_dist = (
            gather_by_index(
                orig_distances, chosen.nonzero(as_tuple=True)[1].view(batch_size_, -1)
            )
            .view(batch_size_, -1, n_points_)
            .min(1)
            .values.sum(-1)
        )
        return -cur_min_dist

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # TODO: check solution validity
        pass

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # TODO: local search
        pass

    @staticmethod
    def get_num_starts(td):
        return td["action_mask"].shape[-1]

    @staticmethod
    def select_start_nodes(td, num_starts):
        num_loc = td["action_mask"].shape[-1]
        return (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
        )
