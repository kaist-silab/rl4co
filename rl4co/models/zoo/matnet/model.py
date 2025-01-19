import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.matnet.policy import MatNetPolicy, MultiStageFFSPPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def select_matnet_policy(env, **policy_params):
    if env.name == "ffsp":
        if env.flatten_stages:
            return MatNetPolicy(env_name=env.name, **policy_params)
        else:
            return MultiStageFFSPPolicy(stage_cnt=env.num_stage, **policy_params)
    else:
        return MatNetPolicy(env_name=env.name, **policy_params)


class MatNet(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module | MatNetPolicy = None,
        num_starts: int = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = select_matnet_policy(env=env, **policy_params)

        # Check if using augmentation and the validation of augmentation function
        if kwargs.get("num_augment", 0) != 0:
            log.warning("MatNet is using augmentation.")
            if (
                kwargs.get("augment_fn") in ["symmetric", "dihedral8"]
                or kwargs.get("augment_fn") is None
            ):
                log.error(
                    "MatNet does not use symmetric or dihedral augmentation. Seeting no augmentation function."
                )
                kwargs["num_augment"] = 0
        else:
            kwargs["num_augment"] = 0

        super(MatNet, self).__init__(
            env=env,
            policy=policy,
            num_starts=num_starts,
            **kwargs,
        )
