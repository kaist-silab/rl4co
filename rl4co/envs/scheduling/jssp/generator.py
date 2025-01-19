import os

from functools import partial

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch.nn.functional import one_hot

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

from .parser import get_max_ops_from_files, read

log = get_pylogger(__name__)


class JSSPGenerator(Generator):
    """Data generator for the Job-Shop Scheduling Problem (JSSP)

    Args:
        num_stage: number of stages
        num_machine: number of machines
        num_job: number of jobs
        min_time: minimum running time of each job on each machine
        max_time: maximum running time of each job on each machine
        flatten_stages: whether to flatten the stages
        one2one_ma_map: whether each machine should have exactly one operation per job (common in jssp benchmark instances)

    Returns:
        A TensorDict with the following key:
            start_op_per_job [batch_size, num_jobs]: first operation of each job
            end_op_per_job [batch_size, num_jobs]: last operation of each job
            proc_times [batch_size, num_machines, total_n_ops]: processing time of ops on machines
            pad_mask [batch_size, total_n_ops]: not all instances have the same number of ops, so padding is used

    """

    def __init__(
        self,
        num_jobs: int = 6,
        num_machines: int = 6,
        min_ops_per_job: int = None,
        max_ops_per_job: int = None,
        min_processing_time: int = 1,
        max_processing_time: int = 99,
        one2one_ma_map: bool = True,
        **unused_kwargs,
    ):
        self.num_jobs = num_jobs
        self.num_mas = num_machines
        # quite common in jssp to have as many ops per job as there are machines
        self.min_ops_per_job = min_ops_per_job or self.num_mas
        self.max_ops_per_job = max_ops_per_job or self.num_mas
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.one2one_ma_map = one2one_ma_map
        if self.one2one_ma_map:
            assert self.min_ops_per_job == self.max_ops_per_job == self.num_mas

        # determines whether to use a fixed number of total operations or let it vary between instances
        # NOTE: due to the way rl4co builds datasets, we need a fixed size here
        self.n_ops_max = self.max_ops_per_job * self.num_jobs

        # FFSP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

    def _simulate_processing_times(self, bs, n_ops_max) -> torch.Tensor:
        if self.one2one_ma_map:
            ops_machine_ids = (
                torch.rand((*bs, self.num_jobs, self.num_mas))
                .argsort(dim=-1)
                .flatten(1, 2)
            )
        else:
            ops_machine_ids = torch.randint(
                low=0,
                high=self.num_mas,
                size=(*bs, n_ops_max),
            )
        ops_machine_adj = one_hot(ops_machine_ids, num_classes=self.num_mas)

        # (bs, max_ops, machines)
        proc_times = torch.ones((*bs, n_ops_max, self.num_mas))
        proc_times = torch.randint(
            self.min_processing_time,
            self.max_processing_time + 1,
            size=(*bs, self.num_mas, n_ops_max),
        )

        # remove proc_times for which there is no corresponding ma-ops connection
        proc_times = proc_times * ops_machine_adj.transpose(1, 2)
        # in JSSP there is only one machine capable to process an operation
        assert (proc_times > 0).sum(1).eq(1).all()
        return proc_times.to(torch.float32)

    def _generate(self, batch_size) -> TensorDict:
        # simulate how many operations each job has
        n_ope_per_job = torch.randint(
            self.min_ops_per_job,
            self.max_ops_per_job + 1,
            size=(*batch_size, self.num_jobs),
        )

        # determine the total number of operations per batch instance (which may differ)
        n_ops_batch = n_ope_per_job.sum(1)  # (bs)
        # determine the maximum total number of operations over all batch instances
        n_ops_max = self.n_ops_max or n_ops_batch.max()

        # generate a mask, specifying which operations are padded
        pad_mask = torch.arange(n_ops_max).unsqueeze(0).expand(*batch_size, -1)
        pad_mask = pad_mask.ge(n_ops_batch[:, None].expand_as(pad_mask))

        # determine the id of the end operation for each job
        end_op_per_job = n_ope_per_job.cumsum(1) - 1

        # determine the id of the starting operation for each job
        # (bs, num_jobs)
        start_op_per_job = torch.cat(
            (
                torch.zeros((*batch_size, 1)).to(end_op_per_job),
                end_op_per_job[:, :-1] + 1,
            ),
            dim=1,
        )

        # simulate processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        proc_times = self._simulate_processing_times(batch_size, n_ops_max)

        td = TensorDict(
            {
                "start_op_per_job": start_op_per_job,
                "end_op_per_job": end_op_per_job,
                "proc_times": proc_times,
                "pad_mask": pad_mask,
            },
            batch_size=batch_size,
        )

        return td


class JSSPFileGenerator(Generator):
    """Data generator for the Job-Shop Scheduling Problem (JSSP) using instance files

    Args:
        path: path to files

    Returns:
        A TensorDict with the following key:
            start_op_per_job [batch_size, num_jobs]: first operation of each job
            end_op_per_job [batch_size, num_jobs]: last operation of each job
            proc_times [batch_size, num_machines, total_n_ops]: processing time of ops on machines
            pad_mask [batch_size, total_n_ops]: not all instances have the same number of ops, so padding is used

    """

    def __init__(self, file_path: str, n_ops_max: int = None, **unused_kwargs):
        self.files = (
            [file_path] if os.path.isfile(file_path) else self.list_files(file_path)
        )
        self.num_samples = len(self.files)

        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

        if len(self.files) > 1:
            n_ops_max = get_max_ops_from_files(self.files)

        ret = map(partial(read, max_ops=n_ops_max), self.files)

        td_list, num_jobs, num_machines, max_ops_per_job = list(zip(*list(ret)))
        num_jobs, num_machines = map(lambda x: x[0], (num_jobs, num_machines))
        max_ops_per_job = max(max_ops_per_job)

        self.td = torch.cat(td_list, dim=0)
        self.num_mas = num_machines
        self.num_jobs = num_jobs
        self.max_ops_per_job = max_ops_per_job
        self.start_idx = 0

    def _generate(self, batch_size: list[int]) -> TensorDict:
        batch_size = np.prod(batch_size)
        if batch_size > self.num_samples:
            log.warning(
                f"Only found {self.num_samples} instance files, but specified dataset size is {batch_size}"
            )
        end_idx = self.start_idx + batch_size
        td = self.td[self.start_idx : end_idx]
        self.start_idx += batch_size
        if self.start_idx >= self.num_samples:
            self.start_idx = 0
        return td

    @staticmethod
    def list_files(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        assert len(files) > 0, "No files found in the specified path"
        return files
