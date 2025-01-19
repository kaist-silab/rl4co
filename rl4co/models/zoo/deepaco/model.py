from typing import Any, Optional

from tensordict import TensorDict
import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.deepaco.policy import DeepACOPolicy


class DeepACO(REINFORCE):
    """Implements DeepACO: https://arxiv.org/abs/2309.14032.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to "no" because the shared baseline is manually implemented.
        train_with_local_search: Whether to train with local search. Defaults to False.
        ls_reward_aug_W: Coefficient to be used for the reward augmentation with the local search. Defaults to 0.95.
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[DeepACOPolicy] = None,
        baseline: REINFORCEBaseline | str = "no",  # Shared baseline is manually implemented
        train_with_local_search: bool = True,
        ls_reward_aug_W: float = 0.95,
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = DeepACOPolicy(
                env_name=env.name, train_with_local_search=train_with_local_search, **policy_kwargs
            )

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

        self.train_with_local_search = train_with_local_search
        self.ls_reward_aug_W = ls_reward_aug_W

    def shared_step(
        self,
        batch: Any,
        batch_idx: int,
        phase: str,
        dataloader_idx: int | None = None,
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            out["loss"] = self.calculate_loss(td, batch, out)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        reward = policy_out["reward"]
        advantage = reward - reward.mean(dim=1, keepdim=True)  # Shared baseline

        if self.train_with_local_search:
            ls_reward = policy_out["ls_reward"]
            ls_advantage = ls_reward - ls_reward.mean(dim=1, keepdim=True)  # Shared baseline
            weighted_advantage = advantage * (1 - self.ls_reward_aug_W) + ls_advantage * self.ls_reward_aug_W
        else:
            weighted_advantage = advantage

        return -(weighted_advantage * policy_out["log_likelihood"]).mean()
