from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.ptrnet.policy import PointerNetworkPolicy


class PointerNetwork(REINFORCE):
    """Pointer Network for neural combinatorial optimization based on REINFORCE
    Based on Vinyals et al. (2015) https://arxiv.org/abs/1506.03134
    Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

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
        policy: PointerNetworkPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        policy = (
            PointerNetworkPolicy(env=env, **policy_kwargs) if policy is None else policy
        )
        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
