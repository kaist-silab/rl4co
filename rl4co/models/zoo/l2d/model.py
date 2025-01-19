from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE, StepwisePPO
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline

from .policy import L2DPolicy, L2DPolicy4PPO


class L2DPPOModel(StepwisePPO):
    """Learning2Dispatch model by Zhang et al. (2020):
    'Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning'

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
        policy: L2DPolicy = None,
        policy_kwargs={},
        **kwargs,
    ):
        assert env.name in [
            "fjsp",
            "jssp",
        ], "L2DModel currently only works for Job-Shop Scheduling Problems"
        if policy is None:
            policy = L2DPolicy4PPO(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, **kwargs)


class L2DModel(REINFORCE):
    """Learning2Dispatch model by Zhang et al. (2020):
    'Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning'

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
        policy: L2DPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        assert env.name in [
            "fjsp",
            "jssp",
        ], "L2DModel currently only works for Job-Shop Scheduling Problems"
        if policy is None:
            policy = L2DPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
