from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index

from .generator import MDCPDPGenerator
from .render import render


class MDCPDPEnv(RL4COEnvBase):
    """Multi Depot Capacitated Pickup and Delivery Problem (MDCPDP) environment.
    One reference to understand the problem could be: Solving the multi-compartment capacitated location routing
    problem with pickup–delivery routes and stochastic demands (https://doi.org/10.1016/j.cie.2015.05.008).
    The environment is made of num_loc + num_depots locations (cities):
        - num_depot depot
        - num_loc / 2 pickup locations
        - num_loc / 2 delivery locations
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location
    The capacity is the maximum number of pickups that the vehicle can carry at the same time

    Observations:
        - locs: locations of the cities [num_loc + num_depot, 2]
        - current_node: current node of the agent [1]
        - to_deliver: if the node is to deliver [1]
        - i: current step [1]
        - action_mask: mask of the available actions [num_loc + num_depot]
        - shape: shape of the observation

    Constraints:
        - The agent cannot visit the same city twice
        - The agent must visit the pickup location before the delivery location
        - The agent must visit the depot at the end of the tour

    Finish Condition:
        - The agent visited all the locations

    Reward:
        - Min-sum: the reward is the negative of the length of the tour
        - Min-max: the reward is the negative of the maximum length of the tour
        - Lateness: the reward is the negative of the cumulate sum of the length of the tour
        - Lateness-square: the reward is the negative of the cumulate sum of the square of the length of the tour

    Args:
        generator: MDCPDPGenerator instance as the data generator
        generator_params: parameters for the generator
        dist_mode: distance mode. One of ["L1", "L2"]
        reward_mode: objective of the problem. One of ["lateness", "lateness_square", "minmax", "minsum"]
        problem_mode: type of the problem. One of ["close", "open"]
        start_mode: type of the start. One of ["order", "random"]
    """

    name = "mdcpdp"

    def __init__(
        self,
        generator: MDCPDPGenerator = None,
        generator_params: dict = {},
        dist_mode: str = "L2",
        reward_mode: str = "lateness",
        problem_mode: str = "close",
        start_mode: str = "order",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MDCPDPGenerator(**generator_params)
        self.generator = generator
        self.dist_mode = dist_mode
        self.reward_mode = reward_mode
        self.problem_mode = problem_mode
        self.start_mode = start_mode
        self.depot_mode = generator.depot_mode
        self._make_spec(self.generator)

        assert self.dist_mode in ["L1", "L2"], "Distance mode (L1/L2) not supported"
        assert self.reward_mode in [
            "lateness",
            "lateness_square",
            "minmax",
            "minsum",
        ], "Objective mode not supported"
        assert self.problem_mode in [
            "close",
            "open",
        ], "Task type (open/close) not supported"
        assert self.start_mode in [
            "order",
            "random",
        ], "Start type (order/random) not supported"

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"].unsqueeze(-1)
        current_depot = td["current_depot"]

        num_depot = td["capacity"].shape[-1]
        num_loc = td["locs"].shape[-2] - num_depot  # no depot
        pd_split_idx = num_loc // 2 + num_depot

        # Pickup and delivery node pair of selected node
        new_to_deliver = (current_node + num_loc // 2) % (num_loc + num_depot)

        # If back to the depot
        back_flag = (current_node < num_depot) & (
            td["available"].gather(-1, current_node) == 0
        )

        # Set available to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node.expand_as(td["action_mask"]), 0
        )

        # Record the to be delivered node
        to_deliver = td["to_deliver"].scatter(
            -1, new_to_deliver.expand_as(td["to_deliver"]), 1
        )

        # Update number of current carry orders
        current_carry = td["current_carry"]
        current_carry += (
            (current_node < pd_split_idx) & (current_node >= num_depot)
        ).long()  # If pickup, add 1
        current_carry -= (current_node >= pd_split_idx).long()  # If delivery, minus 1

        # Update the current depot
        current_depot = td["current_depot"]
        current_depot = torch.where(back_flag, current_node, current_depot)

        # Update the length of current tour
        current_length = td["current_length"]
        prev_loc = gather_by_index(td["locs"], td["current_node"])
        curr_loc = gather_by_index(td["locs"], current_node)
        current_step_length = self.get_distance(prev_loc, curr_loc)

        # If this path is the way between two depods, i.e. open a new route, set the length to 0
        current_step_length = torch.where(
            (current_node < num_depot) & (td["current_node"] < num_depot),
            0,
            current_step_length,
        )

        # If the problem mode is open, the path back to the depot will not be counted
        if self.problem_mode == "open":
            current_step_length = torch.where(
                (current_node < num_depot) & (td["current_node"] >= num_depot),
                0,
                current_step_length,
            )

        # Update the current length
        current_length.scatter_add_(-1, current_depot, current_step_length)

        # Update the arrive time for each city
        arrivetime_record = td["arrivetime_record"]
        arrivetime_record.scatter_(
            -1, current_node, current_length.gather(-1, current_depot)
        )

        # Action is feasible if the node is not visited and is to deliver
        action_mask = available & to_deliver

        # If reach the capacity, only delivery is available
        current_capacity = td["capacity"].gather(-1, current_depot)
        capacity_flag = current_carry >= current_capacity
        action_mask[
            ..., num_depot:pd_split_idx
        ] &= ~capacity_flag  # If reach the capacity, pickup is not available

        # If back to the current depot, this tour is done, set other depots to availbe to start
        # a new tour. Must start from a depot.
        action_mask[..., num_depot:] &= ~back_flag.expand_as(action_mask[..., num_depot:])

        # If back to the depot, other unvisited depots are available
        # if not back to the depot, depots are not available except the current depot
        action_mask[..., :num_depot] &= back_flag.expand_as(action_mask[..., :num_depot])
        action_mask[..., :num_depot].scatter_(-1, current_depot, ~back_flag)

        # If this is the last agent, it has to finish all the left taks
        last_depot_flag = (
            torch.sum(available[..., :num_depot].long(), dim=-1, keepdim=True) == 0
        )
        action_mask[..., :num_depot] &= ~last_depot_flag.expand_as(
            action_mask[..., :num_depot]
        )

        # Update depot mask
        carry_flag = current_carry > 0  # If agent is carrying orders
        action_mask[
            ..., :num_depot
        ] &= ~carry_flag  # If carrying orders, depot is not available

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) == 0

        # If done, the last depot would be always available
        action_mask[..., :num_depot].scatter_(
            -1,
            current_depot,
            action_mask[..., :num_depot].gather(-1, current_depot) | done,
        )

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        # Update step
        td.update(
            {
                "current_node": current_node,
                "current_depot": current_depot,
                "current_carry": current_carry,
                "available": available,
                "to_deliver": to_deliver,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device
        locs = torch.cat((td["depot"], td["locs"]), -2)

        # Record how many depots are visited
        depot_idx = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Pick is 1, deliver is 0 [batch_size, graph_size+1], i.e. [1, 1, ..., 1, 0, ...0]
        to_deliver = torch.cat(
            [
                torch.ones(
                    *batch_size,
                    self.generator.num_loc // 2 + self.generator.num_depot,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros(
                    *batch_size,
                    self.generator.num_loc // 2,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=-1,
        )

        # Current depot index
        if self.start_mode == "random":
            current_depot = torch.randint(
                low=0, high=self.generator.num_depot, size=(*batch_size, 1), device=device
            )
        elif self.start_mode == "order":
            current_depot = torch.zeros(
                (*batch_size, 1), dtype=torch.int64, device=device
            )

        # Current carry order number
        current_carry = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Current length of each depot
        current_length = torch.zeros(
            (*batch_size, self.generator.num_depot), dtype=torch.float32, device=device
        )

        # Arrive time for each city
        arrivetime_record = torch.zeros(
            (*batch_size, self.generator.num_loc + self.generator.num_depot),
            dtype=torch.float32,
            device=device,
        )

        # Cannot visit depot at first step # [0,1...1] so set not available
        available = torch.ones(
            (*batch_size, self.generator.num_loc + self.generator.num_depot),
            dtype=torch.bool,
            device=device,
        )
        action_mask = ~available.contiguous()  # [batch_size, graph_size+1]
        action_mask[..., 0] = 1  # First step is always the depot

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": locs,
                "depot_idx": depot_idx,
                "current_node": current_node,
                "current_depot": current_depot,
                "current_carry": current_carry,
                "current_length": current_length,
                "arrivetime_record": arrivetime_record,
                "capacity": td["capacity"],
                "lateness_weight": td["lateness_weight"],
                "to_deliver": to_deliver,
                "available": available,
                "i": i,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: MDCPDPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            to_deliver=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def get_distance(self, prev_loc, cur_loc):
        # Use L1 norm to calculate the distance for Manhattan distance
        if self.dist_mode == "L1":
            return torch.abs(cur_loc - prev_loc).norm(p=1, dim=-1)
        elif self.dist_mode == "L2":
            return torch.abs(cur_loc - prev_loc).norm(p=2, dim=-1)
        else:
            raise ValueError(f"Invalid distance norm: {self.dist_norm}")

    def _get_reward(self, td: TensorDict, actions) -> TensorDict:
        """Return the rewrad for the current state
        Support modes:
            - minmax: the reward is the maximum length of all agents
            - minsum: the reward is the sum of all agents' length
            - lateness: the reward is the sum of all agents' length plus the lateness with a weight
        Args:
            - actions [batch_size, num_depot+num_locs-1]: the actions taken by the agents
                note that the last city back to depot is not included here
        """
        # Check the validity of the actions
        num_depot = td["capacity"].shape[-1]
        num_loc = td["locs"].shape[-2] - num_depot  # except depot

        # Append the last depot to the end of the actions
        actions = torch.cat([actions, td["current_depot"]], dim=-1)

        # Calculate the reward
        if self.reward_mode == "minmax":
            cost = torch.max(td["current_length"], dim=-1)[0]
        elif self.reward_mode == "minsum":
            cost = torch.sum(td["current_length"], dim=-1)
        elif self.reward_mode == "lateness":
            cost = torch.sum(td["current_length"], dim=(-1))
            lateness = td["arrivetime_record"][..., num_depot + num_loc // 2 :]
            if self.reward_mode == "lateness_square":
                lateness = lateness**2
            lateness = torch.sum(lateness, dim=-1)
            # lateness weight - note that if this is 0, the reward is the same as the cost
            # and if this is 1, the reward is the same as the lateness
            cost = (
                cost * (1 - td["lateness_weight"].squeeze())
                + lateness * td["lateness_weight"].squeeze()
            )
        else:
            raise NotImplementedError(
                f"Invalid reward mode: {self.reward_mode}. Available modes: minmax, minsum, lateness_square, lateness"
            )
        return -cost  # minus for reward

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert True, "Not implemented"

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
