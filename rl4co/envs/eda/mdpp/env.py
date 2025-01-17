from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.eda.dpp.env import DPPEnv
from rl4co.utils.pylogger import get_pylogger

from .generator import MDPPGenerator

log = get_pylogger(__name__)


class MDPPEnv(DPPEnv):
    """Multiple decap placement problem (mDPP) environment
    This is a modified version of the DPP environment where we allow multiple probing ports

    Observations:
        - locations of the probing ports and keepout regions
        - current decap placement
        - remaining decaps

    Constraints:
        - decaps cannot be placed at the probing ports or keepout regions
        - the number of decaps is limited

    Finish Condition:
        - the number of decaps exceeds the limit

    Reward:
        - the impedance suppression at the probing ports

    Args:
        generator: DPPGenerator instance as the data generator
        generator_params: parameters for the generator
        reward_type: reward type, either minmax or meansum
            - minmax: min of the max of the decap scores
            - meansum: mean of the sum of the decap scores

    Note:
        The minmax is more challenging as it requires to find the best decap location
        for the worst case
    """

    name = "mdpp"

    def __init__(
        self,
        generator: MDPPGenerator = None,
        generator_params: dict = {},
        reward_type: str = "minmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MDPPGenerator(**generator_params)
        self.generator = generator

        assert reward_type in [
            "minmax",
            "meansum",
        ], "reward_type must be minmax or meansum"
        self.reward_type = reward_type

        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # Step function is the same as DPPEnv, only masking changes
        return super()._step(td)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Reset function is the same as DPPEnv, only masking changes due to probes
        td_reset = super()._reset(td, batch_size=batch_size)

        # Action mask is 0 if both action_mask (e.g. keepout) and probe are 0
        action_mask = torch.logical_and(td_reset["action_mask"], ~td_reset["probe"])
        # Keepout regions are the inverse of action_mask
        td_reset.update(
            {
                "keepout": ~td_reset["action_mask"],
                "action_mask": action_mask,
            }
        )
        return td_reset

    def _make_spec(self, generator: MDPPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.size**2, 2),
                dtype=torch.float32,
            ),
            probe=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            keepout=Unbounded(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.size**2,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td, actions):
        """We call the reward function with the final sequence of actions to get the reward
        Calling per-step would be very time consuming due to decap simulation
        """
        # We do the operation in a batch
        if len(td.batch_size) == 0:
            td = td.unsqueeze(0)
            actions = actions.unsqueeze(0)

        # Reward calculation is expensive since we need to run decap simulation (not vectorizable)
        reward = torch.stack(
            [
                self._single_env_reward(td_single, action)
                for td_single, action in zip(td, actions)
            ]
        )
        return reward

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert True, "Not implemented"

    def _single_env_reward(self, td, actions):
        """Get reward for single environment. We"""

        list_probe = torch.nonzero(td["probe"]).squeeze()
        scores = torch.zeros_like(list_probe, dtype=torch.float32)
        for i, probe in enumerate(list_probe):
            # Get the decap scores for the probe location
            scores[i] = self._decap_simulator(probe, actions)
        # If minmax, return min of max decap scores else mean
        return scores.min() if self.reward_type == "minmax" else scores.mean()
