import copy

import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.rl import PPO
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AMPPO(PPO):
    """PPO Model based on Proximal Policy Optimization (PPO) with an attention model policy.
    We default to the attention model policy and the Attention Critic Network.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            # we reuse the parameters of the model
            encoder = getattr(policy, "encoder", None)
            if encoder is None:
                raise ValueError("Critic network requires an encoder")
            critic = CriticNetwork(
                copy.deepcopy(encoder).to(next(encoder.parameters()).device),
                **critic_kwargs,
            )

        super().__init__(env, policy, critic, **kwargs)
