from typing import Callable

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MVMoE_POMO(POMO):
    """MVMoE Model for neural combinatorial optimization based on POMO and REINFORCE
    Please refer to Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        moe_kwargs: dict = None,
        **kwargs,
    ):
        if moe_kwargs is None:
            moe_kwargs = {
                "encoder": {
                    "hidden_act": "ReLU",
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
                "decoder": {
                    "light_version": True,
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
            }

        if policy is None:
            policy_kwargs_ = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
                "moe_kwargs": moe_kwargs,
            }
            policy_kwargs.update(policy_kwargs_)
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        # Initialize with the shared baseline
        super(MVMoE_POMO, self).__init__(
            env,
            policy,
            policy_kwargs,
            baseline,
            num_augment,
            augment_fn,
            first_aug_identity,
            feats,
            num_starts,
            **kwargs,
        )


class MVMoE_AM(AttentionModel):
    """MVMoE Model for neural combinatorial optimization based on AM and REINFORCE
    Please refer to Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        moe_kwargs: dict = None,
        **kwargs,
    ):
        if moe_kwargs is None:
            moe_kwargs = {
                "encoder": {
                    "hidden_act": "ReLU",
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
                "decoder": {
                    "light_version": True,
                    "out_bias": False,
                    "num_experts": 4,
                    "k": 2,
                    "noisy_gating": True,
                },
            }

        if policy is None:
            policy_kwargs_ = {
                "moe_kwargs": moe_kwargs,
            }
            policy_kwargs.update(policy_kwargs_)
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        # Initialize with the shared baseline
        super(MVMoE_AM, self).__init__(
            env, policy, baseline, policy_kwargs, baseline_kwargs, **kwargs
        )
