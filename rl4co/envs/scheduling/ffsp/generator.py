import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class FFSPGenerator(Generator):
    """Data generator for the Flow Shop Scheduling Problem (FFSP).

    Args:
        num_stage: number of stages
        num_machine: number of machines
        num_job: number of jobs
        min_time: minimum running time of each job on each machine
        max_time: maximum running time of each job on each machine
        flatten_stages: whether to flatten the stages

    Returns:
        A TensorDict with the following key:
            run_time [batch_size, num_job, num_machine, num_stage]: running time of each job on each machine

    Note:
        - [IMPORTANT] This version of ffsp requires the number of machines in each stage to be the same
    """

    def __init__(
        self,
        num_stage: int = 2,
        num_machine: int = 3,
        num_job: int = 4,
        min_time: int = 2,
        max_time: int = 10,
        flatten_stages: bool = True,
        **unused_kwargs,
    ):
        self.num_stage = num_stage
        self.num_machine = num_machine
        self.num_machine_total = num_machine * num_stage
        self.num_job = num_job
        self.min_time = min_time
        self.max_time = max_time
        self.flatten_stages = flatten_stages

        # FFSP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

    def _generate(self, batch_size) -> TensorDict:
        # Init observation: running time of each job on each machine
        run_time = torch.randint(
            low=self.min_time,
            high=self.max_time,
            size=(*batch_size, self.num_job, self.num_machine_total),
        )

        return TensorDict(
            {
                "run_time": run_time,
            },
            batch_size=batch_size,
        )
