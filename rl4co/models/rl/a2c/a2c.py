import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor
from rl4co.models.rl.reinforce.baselines import CriticBaseline
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class A2C(REINFORCE):
    """Advantage Actor Critic (A2C) algorithm.
    A2C is a variant of REINFORCE where a baseline is provided by a critic network.
    Here we additionally support different optimizers for the actor and the critic.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic network to use for the algorithm
        critic_kwargs: Keyword arguments to pass to the critic network
        actor_optimizer_kwargs: Keyword arguments for the policy (=actor) optimizer
        critic_optimizer_kwargs: Keyword arguments for the critic optimizer. If None, use the same as actor_optimizer_kwargs
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        actor_optimizer_kwargs: dict = {"lr": 1e-4},
        critic_optimizer_kwargs: dict = None,
        **kwargs,
    ):
        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            critic = create_critic_from_actor(policy, **critic_kwargs)

        # The baseline is directly created here, so we eliminate the baseline argument
        kwargs.pop("baseline", None)

        super().__init__(env, policy, baseline=CriticBaseline(critic), **kwargs)
        self.actor_optimizer_kwargs = actor_optimizer_kwargs
        self.critic_optimizer_kwargs = (
            critic_optimizer_kwargs
            if critic_optimizer_kwargs is not None
            else actor_optimizer_kwargs
        )

    def configure_optimizers(self):
        """Configure the optimizers for the policy and the critic network (=baseline)"""
        parameters = [
            {"params": self.policy.parameters(), **self.actor_optimizer_kwargs},
        ] + [{"params": self.baseline.parameters(), **self.critic_optimizer_kwargs}]

        return super().configure_optimizers(parameters)
