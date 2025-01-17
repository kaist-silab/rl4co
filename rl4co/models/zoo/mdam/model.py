from functools import partial

import torch

from torch.utils.data import DataLoader

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import (
    REINFORCEBaseline,
    RolloutBaseline,
    WarmupBaseline,
)
from rl4co.models.zoo.mdam.policy import MDAMPolicy


def rollout(self, model, env, batch_size=64, device="cpu", dataset=None):
    """In this case the reward from the model is [batch, num_paths]
    and the baseline takes the maximum reward from the model as the baseline.
    https://github.com/liangxinedu/MDAM/blob/19b0bf813fb2dbec2fcde9e22eb50e04675400cd/train.py#L38C29-L38C33
    """
    # if dataset is None, use the dataset of the baseline
    dataset = self.dataset if dataset is None else dataset

    model.eval()
    model = model.to(device)

    def eval_model(batch):
        with torch.inference_mode():
            batch = env.reset(batch.to(device))
            return model(batch, env, decode_type="greedy")["reward"].max(1).values

    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    rewards = torch.cat([eval_model(batch) for batch in dl], 0)
    return rewards


class MDAM(REINFORCE):
    """Multi-Decoder Attention Model (MDAM) is a model
    to train multiple diverse policies, which effectively increases the chance of finding
    good solutions compared with existing methods that train only one policy.
    Reference link: https://arxiv.org/abs/2012.10638;
    Implementation reference: https://github.com/liangxinedu/MDAM.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: MDAMPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = MDAMPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

        # Change rollout of baseline to the rollout function
        if isinstance(self.baseline, WarmupBaseline):
            if isinstance(self.baseline.baseline, RolloutBaseline):
                self.baseline.baseline.rollout = partial(rollout, self.baseline.baseline)
        elif isinstance(self.baseline, RolloutBaseline):
            self.baseline.rollout = partial(rollout, self.baseline)

    def calculate_loss(
        self,
        td,
        batch,
        policy_out,
        reward=None,
        log_likelihood=None,
    ):
        """Calculate loss for REINFORCE algorithm.
        Same as in :class:`REINFORCE`, but the bl_val is calculated is simply unsqueezed to match
        the reward shape (i.e., [batch, num_paths])

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        # reward: [batch, num_paths]. Note that the baseline value is the max reward
        # if bl_val is a tensor, unsqueeze it to match the reward shape
        if isinstance(bl_val, torch.Tensor):
            if len(bl_val.shape) > 0:
                bl_val = bl_val.unsqueeze(1)
        advantage = reward - bl_val  # advantage = reward - baseline
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out
