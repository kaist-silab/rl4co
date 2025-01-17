from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import ImprovementEnvBase, RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length

from .generator import PDPGenerator
from .render import render, render_improvement


class PDPEnv(RL4COEnvBase):
    """Pickup and Delivery Problem (PDP) environment.
    The environment is made of num_loc + 1 locations (cities):
        - 1 depot
        - `num_loc` / 2 pickup locations
        - `num_loc` / 2 delivery locations
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location

    Observations:
        - locations of the depot, pickup, and delivery locations
        - current location of the vehicle
        - the remaining locations to deliver
        - the visited locations
        - the current step

    Constraints:
        - the tour starts and ends at the depot
        - each pickup location must be visited before its corresponding delivery location
        - the vehicle cannot visit the same location twice

    Finish Condition:
        - the vehicle has visited all locations

    Reward:
        - (minus) the negative length of the path

    Args:
        generator: PDPGenerator instance as the data generator
        generator_params: parameters for the generator
        force_start_at_depot: whether to force the agent to start at the depot
            If False (default), the agent won't consider the depot, which is added in the `get_reward` method
            If True, the only valid action at the first step is to visit the depot (=0)
    """

    name = "pdp"

    def __init__(
        self,
        generator: PDPGenerator = None,
        generator_params: dict = {},
        force_start_at_depot: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = PDPGenerator(**generator_params)
        self.generator = generator
        self.force_start_at_depot = force_start_at_depot
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"].unsqueeze(-1)

        num_loc = td["locs"].shape[-2] - 1  # except depot

        # Pickup and delivery node pair of selected node
        new_to_deliver = (current_node + num_loc // 2) % (num_loc + 1)

        # Set available to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node.expand_as(td["action_mask"]), 0
        )

        to_deliver = td["to_deliver"].scatter(
            -1, new_to_deliver.expand_as(td["to_deliver"]), 1
        )

        # Action is feasible if the node is not visited and is to deliver
        # action_mask = torch.logical_and(available, to_deliver)
        action_mask = available & to_deliver

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)
        
        # Update step
        td.update(
            {
                "current_node": current_node,
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

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Pick is 1, deliver is 0 [batch_size, graph_size+1], [1,1...1, 0...0]
        to_deliver = torch.cat(
            [
                torch.ones(
                    *batch_size,
                    self.generator.num_loc // 2 + 1,
                    dtype=torch.bool,
                ).to(device),
                torch.zeros(
                    *batch_size,
                    self.generator.num_loc // 2,
                    dtype=torch.bool,
                ).to(device),
            ],
            dim=-1,
        )

        # Masking variables
        available = torch.ones(
            (*batch_size, self.generator.num_loc + 1), dtype=torch.bool
        ).to(device)
        action_mask = torch.ones_like(available) # [batch_size, graph_size+1]
        if self.force_start_at_depot:
            action_mask[..., 1:] = False # can only visit the depot at the first step
        else:
            action_mask = action_mask & to_deliver
            available[..., 0] = False # depot is already visited (during reward calculation)
            action_mask[..., 0] = False # depot is not available to visit

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64).to(device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64).to(device)

        return TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "to_deliver": to_deliver,
                "available": available,
                "i": i,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: PDPGenerator):
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

    @staticmethod
    def _get_reward(td, actions) -> TensorDict:
        # Gather locations in order of tour (add depot since we start and end there)
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],  # depot
                gather_by_index(td["locs"], actions),  # order locations
            ],
            dim=1,
        )
        return -get_tour_length(locs_ordered)

    def check_solution_validity(self, td, actions):
        if not self.force_start_at_depot:
            actions = torch.cat((torch.zeros_like(actions[:, 0:1]), actions), dim=-1)

        assert (
            (torch.arange(actions.size(1), out=actions.data.new()))
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Not visiting all nodes"
        
        # make sure we don't go back to the depot in the middle of the tour
        assert (actions[:, 1:-1] != 0).all(), "Going back to depot in the middle of the tour (not allowed)"

        visited_time = torch.argsort(
            actions, 1
        )  # index of pickup less than index of delivery
        assert (
            visited_time[:, 1 : actions.size(1) // 2 + 1]
            < visited_time[:, actions.size(1) // 2 + 1 :]
        ).all(), "Deliverying without pick-up"

    def get_num_starts(self, td):
        """Only half of the nodes (i.e. pickup nodes) can be start nodes"""
        return (td["locs"].shape[-2] - 1) // 2

    def select_start_nodes(self, td, num_starts):
        """Only nodes from [1 : num_loc // 2 +1] (i.e. pickups) can be selected"""
        num_possible_starts = (td["locs"].shape[-2] - 1) // 2
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_possible_starts
            + 1
        )
        return selected

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)


class PDPRuinRepairEnv(ImprovementEnvBase):
    """Pickup and Delivery Problem (PDP) environment for performing neural ruin-repair search.
    The environment is made of num_loc + 1 locations (cities):
        - 1 depot
        - `num_loc` / 2 pickup locations
        - `num_loc` / 2 delivery locations

    The goal is to search for optimal solutions to pickup and delivery problems by performing a ruin-and-repair neighborhood search on a given initial solution.
    (see MDP described in https://arxiv.org/abs/2204.11399)

    The condition is that at each step, the visited solutions must be feasible,
    maintaining the sequence of visiting the pickup location before its corresponding delivery location.

    Observations:
        - locations of the depot, pickup, and delivery locations
        - current solution to be improved
        - historical decisions
        - the current step

    Constraints:
        - the tour starts and ends at the depot
        - each pickup location must be visited before its corresponding delivery location
        - the vehicle cannot visit the same location twice

    Finish Condition:
        - None

    Reward:
        - the immediate reduced cost over the current best-so-far solution
        (see MDP described in https://arxiv.org/abs/2204.11399)

    Args:
        num_loc: number of locations (cities) in the TSP
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "pdp_ruin_repair"

    def __init__(
        self,
        generator: PDPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = PDPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict, solution_to=None) -> TensorDict:
        # get state information from td
        solution_best = td["rec_best"]
        locs = td["locs"]
        cost_bsf = td["cost_bsf"]
        action_record = td["action_record"]
        bs, gs = solution_best.size()

        # perform local_operator
        if solution_to is None:
            action = td["action"]
            solution = td["rec_current"]
            next_rec = self._local_operator(solution, action)
        else:
            next_rec = solution_to.clone()
        new_obj = self.get_costs(locs, next_rec)

        # compute reward and update best-so-far solutions
        now_bsf = torch.where(new_obj < cost_bsf, new_obj, cost_bsf)
        reward = cost_bsf - now_bsf
        index = reward > 0.0
        solution_best[index] = next_rec[index].clone()

        # reset visited_time
        visited_time = td["visited_time"] * 0
        pre = torch.zeros((bs), device=visited_time.device).long()
        arange = torch.arange(bs)
        for i in range(gs):
            current_nodes = next_rec[arange, pre]
            visited_time[arange, current_nodes] = i + 1
            pre = current_nodes
        visited_time = visited_time.long()

        # update action record
        if solution_to is None:
            action_record[:, :-1] = action_record[:, 1:]
            action_record[:, -1] *= 0
            action_record[torch.arange(bs), -1, action[:, 0]] = 1

        # Update step
        td.update(
            {
                "cost_current": new_obj,
                "cost_bsf": now_bsf,
                "rec_current": next_rec,
                "rec_best": solution_best,
                "visited_time": visited_time,
                "action_record": action_record,
                "i": td["i"] + 1 if solution_to is None else td["i"],
                "reward": reward,
            }
        )

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)
        current_rec = self.generator._get_initial_solutions(locs).to(device)
        obj = self.get_costs(locs, current_rec)

        # get index according to the solutions in the linked list data structure
        bs = batch_size[0]
        seq_length = self.generator.num_loc + 1
        visited_time = torch.zeros((bs, seq_length)).to(device)
        pre = torch.zeros((bs)).to(device).long()
        arange = torch.arange(bs)
        for i in range(seq_length):
            current_nodes = current_rec[arange, pre]
            visited_time[arange, current_nodes] = i + 1
            pre = current_nodes
        visited_time = visited_time.long()

        # get action record and step i
        i = torch.zeros((*batch_size, 1), dtype=torch.int64).to(device)
        action_record = (
            torch.zeros((bs, seq_length, seq_length // 2))
            if self.training
            else torch.zeros((bs, seq_length // 2, seq_length // 2))
        )

        return TensorDict(
            {
                "locs": locs,
                "cost_current": obj,
                "cost_bsf": obj.clone(),
                "rec_current": current_rec,
                "rec_best": current_rec.clone(),
                "visited_time": visited_time,
                "action_record": action_record,
                "i": i,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def _local_operator(solution, action):
        # get info
        pair_index = action[:, 0].view(-1, 1) + 1
        first = action[:, 1].view(-1, 1)
        second = action[:, 2].view(-1, 1)
        rec = solution.clone()
        bs, gs = rec.size()

        # fix connection for pairing node
        argsort = rec.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec.gather(1, pair_index)
        rec.scatter_(1, pre_pairfirst, post_pairfirst)
        rec.scatter_(1, pair_index, pair_index)

        argsort = rec.argsort()

        pre_pairsecond = argsort.gather(1, pair_index + gs // 2)
        post_pairsecond = rec.gather(1, pair_index + gs // 2)

        rec.scatter_(1, pre_pairsecond, post_pairsecond)

        # fix connection for pairing node
        post_second = rec.gather(1, second)
        rec.scatter_(1, second, pair_index + gs // 2)
        rec.scatter_(1, pair_index + gs // 2, post_second)

        post_first = rec.gather(1, first)
        rec.scatter_(1, first, pair_index)
        rec.scatter_(1, pair_index, post_first)

        return rec

    def _make_spec(self, generator: PDPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            cost_current=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            cost_bsf=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            rec_current=Unbounded(
                shape=(self.generator.num_loc + 1),
                dtype=torch.int64,
            ),
            rec_best=Unbounded(
                shape=(self.generator.num_loc + 1),
                dtype=torch.int64,
            ),
            visited_time=Unbounded(
                shape=(self.generator.num_loc + 1, self.generator.num_loc + 1),
                dtype=torch.int64,
            ),
            action_record=Unbounded(
                shape=(self.generator.num_loc + 1, self.generator.num_loc + 1),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(3,),
            dtype=torch.int64,
            low=0,
            high=self.generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def check_solution_validity(self, td, actions=None):
        # The function can be called by the agent to check the validity of the best found solution
        # Note that the args actions are not used in improvement method.

        solution = td["rec_best"]
        batch_size, graph_size = solution.size()

        assert (
            torch.arange(graph_size, out=solution.data.new())
            .view(1, -1)
            .expand_as(solution)
            == solution.data.sort(1)[0]
        ).all(), "Not visiting all nodes"

        visited_time = torch.zeros((batch_size, graph_size), device=self.device)
        pre = torch.zeros(batch_size, device=self.device).long()
        arange = torch.arange(batch_size)
        for i in range(graph_size):
            visited_time[arange, solution[arange, pre]] = i + 1
            pre = solution[arange, pre]

        assert (
            visited_time[:, 1 : graph_size // 2 + 1]
            < visited_time[:, graph_size // 2 + 1 :]
        ).all(), "Deliverying without pick-up"

    @staticmethod
    def get_mask(selected_node, td):
        # return mask that is 1 if the corresponding action is feasible, 0 otherwise

        visited_time = td["visited_time"]
        bs, gs = visited_time.size()
        visited_time = visited_time % gs
        arange = torch.arange(bs)

        visited_order_map = visited_time.view(bs, gs, 1) > visited_time.view(bs, 1, gs)
        mask = visited_order_map.clone()
        mask[arange, selected_node.view(-1)] = True
        mask[arange, selected_node.view(-1) + gs // 2] = True
        mask[arange, :, selected_node.view(-1)] = True
        mask[arange, :, selected_node.view(-1) + gs // 2] = True

        bs, gs, _ = visited_order_map.size()

        return ~mask

    @classmethod
    def _random_action(cls, td):
        batch_size, graph_size = td["rec_best"].size()
        selected_node = (
            (torch.rand(batch_size, 1) * graph_size // 2) % (graph_size // 2)
        ).long()
        mask = cls.get_mask(selected_node + 1, td)
        logits = torch.rand(batch_size, graph_size, graph_size)
        logits[~mask] = -1e20
        prob = torch.softmax(logits.view(batch_size, -1), -1)
        sample = prob.multinomial(1)
        action = torch.cat(
            (selected_node, sample // (graph_size), sample % (graph_size)), -1
        )
        td["action"] = action
        return action

    @classmethod
    def render(cls, td: TensorDict, actions: torch.Tensor = None, ax=None):
        solution_current = cls.get_current_solution(td)
        solution_best = cls.get_best_solution(td)
        return render_improvement(td, solution_current, solution_best)
