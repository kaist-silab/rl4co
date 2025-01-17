import itertools

from math import factorial
from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.dataset import FastTdDataset
from rl4co.envs.common.base import RL4COEnvBase

from .generator import FFSPGenerator


class FFSPEnv(RL4COEnvBase):
    """Flexible Flow Shop Problem (FFSP) environment.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.

    Observations:
        - time index
        - sub time index
        - batch index
        - machine index
        - schedule
        - machine wait step
        - job location
        - job wait step
        - job duration

    Constraints:
        - each job has to be processed on each machine in a specific order
        - the machine has to be available to process the job
        - the job has to be available to be processed

    Finish Condition:
        - all jobs are scheduled

    Reward:
        - (minus) the makespan of the schedule

    Args:
        generator: FFSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "ffsp"

    def __init__(
        self,
        generator: FFSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(check_solution=False, dataset_cls=FastTdDataset, **kwargs)
        if generator is None:
            generator = FFSPGenerator(**generator_params)
        self.generator = generator

        self.num_stage = generator.num_stage
        self.num_machine = generator.num_machine
        self.num_job = generator.num_job
        self.num_machine_total = generator.num_machine_total
        self.tables = None
        self.step_cnt = None
        self.flatten_stages = generator.flatten_stages

        self._make_spec(generator)

    def get_num_starts(self, td):
        return factorial(self.num_machine)

    def select_start_nodes(self, td, num_starts):
        self.tables.augment_machine_tables(num_starts)
        selected = torch.full((num_starts * td.size(0),), self.num_job)
        return selected

    def _move_to_next_machine(self, td):
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)

        time_idx = td["time_idx"]
        machine_idx = td["machine_idx"]
        sub_time_idx = td["sub_time_idx"]

        machine_wait_step = td["machine_wait_step"]
        job_wait_step = td["job_wait_step"]
        job_location = td["job_location"]

        ready = torch.flatten(td["done"])
        idx = torch.flatten(batch_idx)
        # select minibatch instances that need updates (are not done)
        idx = idx[~ready]

        while ~ready.all():
            # increment the stage-machine counter
            new_sub_time_idx = sub_time_idx[idx] + 1
            # increment time if all machines-stage combinations have been candidates
            step_time_required = new_sub_time_idx == self.num_machine_total
            time_idx[idx] += step_time_required.long()
            # in this case set the machine-stage counter to zero again
            new_sub_time_idx[step_time_required] = 0
            # update machine-stage counter
            sub_time_idx[idx] = new_sub_time_idx
            # determine current machine candidate
            new_machine_idx = self.tables.get_machine_index(idx, new_sub_time_idx)
            machine_idx[idx] = new_machine_idx

            # decrease machine wait time by 1 if instance transitioned to new time step
            machine_wait_steps = machine_wait_step[idx, :]
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            machine_wait_step[idx, :] = machine_wait_steps

            # decrease job wait time by 1 if instance transitioned to new time step
            job_wait_steps = job_wait_step[idx, :]
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            job_wait_step[idx, :] = job_wait_steps
            # machine is ready if its wait time is zero
            machine_ready = machine_wait_step[idx, new_machine_idx] == 0
            # job is ready if the current stage matches the stage of the job and
            # its wait time is zero (no operation of previous stage is in process)
            new_stage_idx = self.tables.get_stage_index(new_sub_time_idx)
            job_ready_1 = job_location[idx, : self.num_job] == new_stage_idx[:, None]
            job_ready_2 = job_wait_step[idx, : self.num_job] == 0
            job_ready = (job_ready_1 & job_ready_2).any(dim=-1)
            # instance ready if at least one job and the current machine are ready
            ready = machine_ready & job_ready
            assert ready.shape == idx.shape
            idx = idx[~ready]

        return td.update(
            {
                "time_idx": time_idx,
                "sub_time_idx": sub_time_idx,
                "machine_idx": machine_idx,
                "machine_wait_step": machine_wait_step,
                "job_wait_step": job_wait_step,
            }
        )

    def pre_step(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        sub_time_idx = td["sub_time_idx"]
        # update machine index
        td["machine_idx"] = self.tables.get_machine_index(batch_idx, sub_time_idx)
        # update action mask and stage machine indx
        td = self._update_step_state(td)
        # perform some checks
        assert (td["stage_idx"] == 0).all(), "call pre_step only at beginning of env"
        assert torch.all(td["stage_machine_idx"] == td["machine_idx"])
        # return updated td
        return td

    def _update_step_state(self, td):
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)

        sub_time_idx = td["sub_time_idx"]
        job_location = td["job_location"]
        job_wait_step = td["job_wait_step"]
        if len(td["done"].shape) == 2:
            done = td["done"].squeeze(1)
        else:
            done = td["done"]

        # update stage
        stage_idx = self.tables.get_stage_index(sub_time_idx)
        stage_machine_idx = self.tables.get_stage_machine_index(batch_idx, sub_time_idx)

        job_loc = job_location[:, : self.num_job]
        job_wait_time = job_wait_step[:, : self.num_job]
        # determine if job can be scheduled in current stage
        # (i.e. previous stages are completed)
        job_in_stage = job_loc == stage_idx[:, None]
        job_not_waiting = job_wait_time == 0
        # job can be scheduled if in current stage and not waiting
        job_available = job_in_stage & job_not_waiting
        # determine instance for which waiting is allowed. This is the case if either
        # 1.) any of its jobs need to be scheduled in a previous stage,
        # 2.) any of the jobs wait for an operation of the preceeding stage to finish
        # 3.) the instance is done.
        job_in_previous_stages = (job_loc < stage_idx[:, None]).any(dim=-1)
        job_waiting_in_stage = (job_in_stage & (job_wait_time > 0)).any(dim=-1)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + done

        job_enable = torch.cat((job_available, wait_allowed[:, None]), dim=-1)
        job_mask = torch.full_like(td["action_mask"], 0).masked_fill(job_enable, 1)
        assert torch.logical_or((job_mask[:, :-1].sum(1) > 0), done).all()
        return td.update(
            {
                "action_mask": job_mask,
                "stage_idx": stage_idx,
                "stage_machine_idx": stage_machine_idx,
            }
        )

    def _step(self, td: TensorDict) -> TensorDict:
        self.step_cnt += 1
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)

        # job_idx is the action from the model
        job_idx = td["action"]
        time_idx = td["time_idx"]
        machine_idx = td["machine_idx"]

        # increment the operation counter of the selected job
        td["job_location"][batch_idx, job_idx] += 1
        # td["job_location"][:, :-1].clip_(0, self.num_stage)
        # assert (td["job_location"][:, : self.num_job] <= self.num_stage).all()
        # insert start time of the selected job in the schedule
        td["schedule"][batch_idx, machine_idx, job_idx] = time_idx
        # get the duration of the selected job
        job_length = td["job_duration"][batch_idx, job_idx, machine_idx]
        # set the number of time steps until the selected machine is available again
        td["machine_wait_step"][batch_idx, machine_idx] = job_length

        # set the number of time steps until the next operation of the job can be started
        td["job_wait_step"][batch_idx, job_idx] = job_length
        # determine whether all jobs are scheduled
        td["done"] = (td["job_location"][:, : self.num_job] == self.num_stage).all(dim=-1)

        if td["done"].all():
            pass
        else:
            td = self._move_to_next_machine(td)
            td = self._update_step_state(td)

        if td["done"].all():
            # determine end times of ops by adding the durations to their start times
            end_schedule = td["schedule"] + td["job_duration"].permute(0, 2, 1)
            # exclude dummy job and determine the makespan per job
            end_time_max, _ = end_schedule[:, :, : self.num_job].max(dim=-1)
            # determine the max makespan of all jobs
            end_time_max, _ = end_time_max.max(dim=-1)
            reward = -end_time_max.to(torch.float32)
            td.set("reward", reward)

        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """
        Args:

        Returns:
            - stage_table [batch_size, num_stage * num_machine]
            - machine_table [batch_size, num_machine * num_stage]
            - stage_machine_idx [batch_size, num_stage * num_machine]
            - time_idx [batch_size]
            - sub_time_idx [batch_size]
            - batch_idx [batch_size]
            - machine_idx [batch_size]
            - schedule [batch_size, num_machine_total, num_job+1]
            - machine_wait_step [batch_size, num_machine_total]
            - job_location [batch_size, num_job+1]
            - job_wait_step [batch_size, num_job+1]
            - job_duration [batch_size, num_job+1, num_machine * num_stage]
        """
        device = td.device

        self.step_cnt = 0
        self.tables = IndexTables(self)
        # reset tables to undo the augmentation
        # self.tables._reset(device=self.device)
        self.tables.set_bs(batch_size[0])

        # Init index record tensor
        time_idx = torch.zeros(size=(*batch_size,), dtype=torch.long, device=device)
        sub_time_idx = torch.zeros(size=(*batch_size,), dtype=torch.long, device=device)

        # Scheduling status information
        schedule = torch.full(
            size=(*batch_size, self.num_machine_total, self.num_job + 1),
            dtype=torch.long,
            device=device,
            fill_value=-999999,
        )
        machine_wait_step = torch.zeros(
            size=(*batch_size, self.num_machine_total),
            dtype=torch.long,
            device=device,
        )
        job_location = torch.zeros(
            size=(*batch_size, self.num_job + 1),
            dtype=torch.long,
            device=device,
        )
        job_wait_step = torch.zeros(
            size=(*batch_size, self.num_job + 1),
            dtype=torch.long,
            device=device,
        )
        job_duration = torch.empty(
            size=(*batch_size, self.num_job + 1, self.num_machine * self.num_stage),
            dtype=torch.long,
            device=device,
        )
        job_duration[..., : self.num_job, :] = td["run_time"]
        job_duration[..., self.num_job, :] = 0

        # Finish status information
        reward = torch.full(
            size=(*batch_size,),
            dtype=torch.float32,
            device=device,
            fill_value=float("-inf"),
        )
        done = torch.full(
            size=(*batch_size,),
            dtype=torch.bool,
            device=device,
            fill_value=False,
        )

        action_mask = torch.ones(
            size=(*batch_size, self.num_job + 1), dtype=bool, device=device
        )
        action_mask[..., -1] = 0

        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        stage_idx = self.tables.get_stage_index(sub_time_idx)
        machine_idx = self.tables.get_machine_index(batch_idx, sub_time_idx)
        stage_machine_idx = self.tables.get_stage_machine_index(batch_idx, sub_time_idx)

        return TensorDict(
            {
                # Index information
                "stage_idx": stage_idx,
                "time_idx": time_idx,
                "sub_time_idx": sub_time_idx,
                "machine_idx": machine_idx,
                "stage_machine_idx": stage_machine_idx,
                # Scheduling status information
                "schedule": schedule,
                "machine_wait_step": machine_wait_step,
                "job_location": job_location,
                "job_wait_step": job_wait_step,
                "job_duration": job_duration,
                # Finish status information
                "reward": reward,
                "done": done,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: FFSPGenerator):
        self.observation_spec = Composite(
            time_idx=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            sub_time_idx=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            batch_idx=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            machine_idx=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            schedule=Unbounded(
                shape=(generator.num_machine_total, generator.num_job + 1),
                dtype=torch.int64,
            ),
            machine_wait_step=Unbounded(
                shape=(generator.num_machine_total),
                dtype=torch.int64,
            ),
            job_location=Unbounded(
                shape=(generator.num_job + 1),
                dtype=torch.int64,
            ),
            job_wait_step=Unbounded(
                shape=(generator.num_job + 1),
                dtype=torch.int64,
            ),
            job_duration=Unbounded(
                shape=(
                    generator.num_job + 1,
                    generator.num_machine * generator.num_stage,
                ),
                dtype=torch.int64,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_machine_total,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td, actions) -> TensorDict:
        return td["reward"]


class IndexTables:
    def __init__(self, env: FFSPEnv):
        self.stage_table = torch.arange(
            env.num_stage, dtype=torch.long, device=env.device
        ).repeat_interleave(env.num_machine)

        # determine the increment of machine ids between stages, i.e. [0,4,8]
        # for instances with 4 machines and three stages
        start_sub_ids = torch.tensor(
            list(range(0, env.num_machine * env.num_stage, env.num_machine)),
            dtype=torch.long,
            device=env.device,
        ).repeat_interleave(env.num_machine)
        # generate all possible permutations of the machine ids and add the stage increment to it
        # (num_permutations, total_machines)
        permutations = torch.tensor(
            list(itertools.permutations(list(range(env.num_machine)))),
            dtype=torch.long,
            device=env.device,
        ).repeat(1, env.num_stage)
        self.machine_table = permutations + start_sub_ids[None]

        if env.flatten_stages:
            # when flatting stages, every machine in each stage is treated as a distinct entity (no shared embeddings)
            # Therefore, all machine need a unique index which is the same as the machine table
            self.stage_machine_table = self.machine_table
        else:
            # when we do not flatten the stages, machines of different stages with the same subtime index
            # share an embedding. In this case, they need the same index (i.e. leave out the stage increment)
            self.stage_machine_table = permutations

    def set_bs(self, bs):
        self.bs = bs

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, idx, sub_time_idx):
        pomo_idx = idx // self.bs

        return self.machine_table[pomo_idx, sub_time_idx]

    def get_stage_machine_index(self, idx, sub_time_idx):
        pomo_idx = idx // self.bs
        return self.stage_machine_table[pomo_idx, sub_time_idx]
