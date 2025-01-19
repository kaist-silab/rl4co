from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.ham.policy import HeterogeneousAttentionModelPolicy


class HeterogeneousAttentionModel(REINFORCE):
    """Heterogenous Attention Model for solving the Pickup and Delivery Problem based on
    REINFORCE: https://arxiv.org/abs/2110.02634.

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
        policy: HeterogeneousAttentionModelPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        assert (
            env.name == "pdp"
        ), "HeterogeneousAttentionModel only works for PDP (Pickup and Delivery Problem)"
        if policy is None:
            policy = HeterogeneousAttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
