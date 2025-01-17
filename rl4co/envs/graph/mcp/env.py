from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import MCPGenerator

log = get_pylogger(__name__)


class MCPEnv(RL4COEnvBase):
    """Maximum Coverage Problem (MCP) environment
    At each step, the agent chooses a set. The reward is 0 unless enough number of sets are chosen.
    The reward is the total weights of the covered items (i.e., items in any chosen set).

    Observations:
        - the weights of items
        - the membership of items in sets
        - the number of sets to choose

    Constraints:
        - the given number of sets must be chosen

    Finish condition:
        - the given number of sets are chosen

    Reward:
        - the total weights of the covered items (i.e., items in any chosen set)

    Args:
        generator: MCPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "mcp"

    def __init__(
        self,
        generator: MCPGenerator = None,
        generator_params: dict = {},
        check_solution=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MCPGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, 1]; the set to be chosen in each instance
        batch_size = td["action"].shape[0]
        selected = td["action"]

        # Update set selection status
        chosen = td["chosen"].clone()  # (batch_size, n_sets)
        chosen[torch.arange(batch_size).to(td.device), selected] = True

        # We are done if we choose enough sets
        done = td["i"] >= (td["n_sets_to_choose"] - 1)

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        remaining_sets = ~chosen  # (batch_size, n_sets)

        chosen_membership = chosen.unsqueeze(-1) * td["membership"]
        chosen_membership_nonzero = chosen_membership.nonzero()
        remaining_membership = remaining_sets.unsqueeze(-1) * td["membership"]

        batch_indices, set_indices, item_indices = chosen_membership_nonzero.T
        chosen_items_indices = chosen_membership[
            batch_indices, set_indices, item_indices
        ].long()

        batch_size, n_items = td["weights"].shape

        # We have batch_indices and chosen_items_indices
        # chosen_items: (batch_size, n_items)
        # for each i, chosen_items[batch_size[i], chosen_items_indices[i]] += 1
        chosen_items = torch.zeros(batch_size, n_items + 1, device=td.device)
        chosen_items[batch_indices, chosen_items_indices] += 1
        chosen_items = chosen_items[:, 1:]  # Remove the first column (invalid zeros)

        # chosen_item[i, j] > 0 means item j is chosen in batch i
        covered_items = (chosen_items > 0).float()  # (batch_size, n_items)
        remaining_items = 1.0 - covered_items  # (batch_size, n_items)

        # We cannot choose the already-chosen sets
        action_mask = ~chosen

        td.update(
            {
                "membership": remaining_membership,  # (batch_size, n_sets, max_size)
                "weights": td["weights"] * remaining_items,  # (batch_size, n_items)
                "chosen": chosen,
                "i": td["i"] + 1,
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
                # given information; constant for each given instance
                "orig_membership": td["membership"],  # (batch_size, n_sets, max_size)
                "membership": td["membership"],  # (batch_size, n_sets, max_size)
                "orig_weights": td["weights"],  # (batch_size, n_items)
                "weights": td["weights"],  # (batch_size, n_items)
                "n_sets_to_choose": td["n_sets_to_choose"],  # (batch_size, 1)
                # states changed by actions
                "chosen": torch.zeros(
                    *td["membership"].shape[:-1], dtype=torch.bool, device=td.device
                ),  # each entry is binary; 1 iff the corresponding set is chosen
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),  # the number of sets we have chosen
                "action_mask": torch.ones(
                    *td["membership"].shape[:-1], dtype=torch.bool, device=td.device
                ),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: MCPGenerator):
        # TODO: make spec
        pass

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        membership = td[
            "orig_membership"
        ]  # (batch_size, n_sets, max_size); membership[i, j] = the items in set j in batch i (with 0 padding)
        weights = td["orig_weights"]  # (batch_size, n_items)
        chosen_sets = td["chosen"]  # (batch_size, n_set); 1 if chosen, 0 otherwise

        chosen_membership = chosen_sets.unsqueeze(-1) * membership
        chosen_membership_nonzero = chosen_membership.nonzero()

        batch_indices, set_indices, item_indices = chosen_membership_nonzero.T
        chosen_items_indices = chosen_membership[
            batch_indices, set_indices, item_indices
        ].long()

        batch_size, n_items = weights.shape

        # We have batch_indices and chosen_items_indices
        # chosen_items: (batch_size, n_items)
        # For each i, chosen_items[batch_size[i], chosen_items_indices[i]] += 1
        chosen_items = torch.zeros(batch_size, n_items + 1, device=td.device)
        chosen_items[batch_indices, chosen_items_indices] += 1
        chosen_items = chosen_items[:, 1:]  # remove the first column

        # chosen_item[i, j] > 0 means item j is chosen in batch i
        chosen_items = (chosen_items > 0).float()
        # Compute the total weights of chosen items
        chosen_weights = torch.sum(chosen_items * weights, dim=-1)

        return chosen_weights

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
        num_sets = td["action_mask"].shape[-1]
        return (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_sets
        )
