import os
import zipfile
from typing import Union, Callable

import torch
import numpy as np

from robust_downloader import download
from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)


class SMTWTPGenerator(Generator):
    """Data generator for the Single Machine Total Weighted Tardiness Problem (SMTWTP) environment
    
    Args:
        num_job: number of jobs
        min_time_span: lower bound of jobs' due time. By default, jobs' due time is uniformly sampled from (min_time_span, max_time_span)
        max_time_span: upper bound of jobs' due time. By default, it will be set to num_job / 2
        min_job_weight: lower bound of jobs' weights. By default, jobs' weights are uniformly sampled from (min_job_weight, max_job_weight)
        max_job_weight: upper bound of jobs' weights
        min_process_time: lower bound of jobs' process time. By default, jobs' process time is uniformly sampled from (min_process_time, max_process_time)
        max_process_time: upper bound of jobs' process time
    
    Returns:
        A TensorDict with the following key:
            job_due_time [batch_size, num_job + 1]: the due time of each job
            job_weight [batch_size, num_job + 1]: the weight of each job
            job_process_time [batch_size, num_job + 1]: the process time of each job
    """
    def __init__(
        self,
        num_job: int = 10,
        min_time_span: float = 0,
        max_time_span: float = None, # will be set to num_job / 2 by default. In DeepACO, it is set to num_job, which would be too simple
        min_job_weight: float = 0,
        max_job_weight: float = 1,
        min_process_time: float = 0,
        max_process_time: float = 1,
        **unused_kwargs
    ):
        self.num_job = num_job
        self.min_time_span = min_time_span
        self.max_time_span = num_job / 2 if max_time_span is None else max_time_span
        self.min_job_weight = min_job_weight
        self.max_job_weight = max_job_weight
        self.min_process_time = min_process_time
        self.max_process_time = max_process_time

        # SMTWTP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

    def _generate(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        # Sampling according to Ye et al. (2023)
        job_due_time = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_time_span, self.max_time_span)
        )
        job_weight = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_job_weight, self.max_job_weight)
        )
        job_process_time = (
            torch.FloatTensor(*batch_size, self.num_job + 1)
            .uniform_(self.min_process_time, self.max_process_time)
        )

        # Rollouts begin at dummy node 0, whose features are set to 0
        job_due_time[:, 0] = 0
        job_weight[:, 0] = 0
        job_process_time[:, 0] = 0

        return TensorDict(
            {
                "job_due_time": job_due_time,
                "job_weight": job_weight,
                "job_process_time": job_process_time,
            },
            batch_size=batch_size,
        )
