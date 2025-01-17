from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import ImprovementEnvBase, RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import TSPGenerator

try:
    from .local_search import local_search
except ImportError:
    # In case some dependencies are not installed (e.g., numba)
    local_search = None
from .render import render, render_improvement

log = get_pylogger(__name__)


class TSPEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - locations of each customer.
        - the current location of the vehicle.

    Constraints:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: TSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "tsp"

    def __init__(
        self,
        generator: TSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = TSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = td["locs"]

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: TSPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1))
        self.done_spec = Unbounded(shape=(1), dtype=torch.bool)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    def replace_selected_actions(
        self,
        cur_actions: torch.Tensor,
        new_actions: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace selected current actions with updated actions based on `selection_mask`.

        Args:
            cur_actions [batch_size, num_loc]
            new_actions [batch_size, num_loc]
            selection_mask [batch_size,]
        """
        cur_actions[selection_mask] = new_actions[selection_mask]
        return cur_actions

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        assert (
            local_search is not None
        ), "Cannot import local_search module. Check if `numba` is installed."
        return local_search(td, actions, **kwargs)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)


class TSPkoptEnv(ImprovementEnvBase):
    """Traveling Salesman Problem (PDP) environment for performing the neural k-opt search.

    The goal is to search for optimal solutions to TSP by performing a k-opt neighborhood search on a given initial solution.

    Observations:
        - locations of each customer
        - current solution to be improved
        - the current step

    Constraints:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - None

    Reward:
        - the immediate reduced cost over the current best-so-far solution

    Args:
        generator: TSPGenerator instance as the data generator
        generator_params: parameters for the generator
        k_max: the maximum k value for k-opt:
            if k_max==2, the MDP in DACT(https://arxiv.org/abs/2110.02544) is used;
            if k_max>2, the MDP in NeuOpt(https://arxiv.org/abs/2310.18264) is used;
    """

    name = "tsp_kopt"

    def __init__(
        self,
        generator: TSPGenerator = None,
        generator_params: dict = {},
        k_max: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = TSPGenerator(**generator_params)
        self.generator = generator
        self.k_max = k_max
        self.two_opt_mode = k_max == 2
        self._make_spec(self.generator)

    def _step(self, td: TensorDict, solution_to=None) -> TensorDict:
        # get state information from td
        solution_best = td["rec_best"]
        locs = td["locs"]
        cost_bsf = td["cost_bsf"]
        bs, gs = solution_best.size()

        # perform loca_operator
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

        # Update step
        td.update(
            {
                "cost_current": new_obj,
                "cost_bsf": now_bsf,
                "rec_current": next_rec,
                "rec_best": solution_best,
                "visited_time": visited_time,
                "i": td["i"] + 1 if solution_to is None else td["i"],
                "reward": reward,
            }
        )

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        locs = td["locs"]
        current_rec = self.generator._get_initial_solutions(locs).to(device)

        obj = self.get_costs(locs, current_rec)

        # get index according to the solutions in the linked list data structure
        bs = batch_size[0]
        seq_length = self.generator.num_loc
        visited_time = torch.zeros((bs, seq_length)).to(device)
        pre = torch.zeros((bs)).to(device).long()
        arange = torch.arange(bs)
        for i in range(seq_length):
            current_nodes = current_rec[arange, pre]
            visited_time[arange, current_nodes] = i + 1
            pre = current_nodes
        visited_time = visited_time.long()

        i = torch.zeros((*batch_size, 1), dtype=torch.int64).to(device)

        return TensorDict(
            {
                "locs": locs,
                "cost_current": obj,
                "cost_bsf": obj.clone(),
                "rec_current": current_rec,
                "rec_best": current_rec.clone(),
                "visited_time": visited_time,
                "i": i,
            },
            batch_size=batch_size,
        )

    def _local_operator(self, solution, action):
        rec = solution.clone()

        if self.two_opt_mode:
            # get actions
            first = action[:, 0].view(-1, 1)
            second = action[:, 1].view(-1, 1)

            # fix connection for first node
            argsort = solution.argsort()
            pre_first = argsort.gather(1, first)
            pre_first = torch.where(pre_first != second, pre_first, first)
            rec.scatter_(1, pre_first, second)

            # fix connection for second node
            post_second = solution.gather(1, second)
            post_second = torch.where(post_second != first, post_second, second)
            rec.scatter_(1, first, post_second)

            # reverse loop:
            cur = first
            for i in range(self.generator.num_loc):
                cur_next = solution.gather(1, cur)
                rec.scatter_(
                    1, cur_next, torch.where(cur != second, cur, rec.gather(1, cur_next))
                )
                cur = torch.where(cur != second, cur_next, cur)

            rec_next = rec

        else:
            # action bs * (K_index, K_from, K_to)
            selected_index = action[:, : self.k_max]
            left = action[:, self.k_max : 2 * self.k_max]
            right = action[:, 2 * self.k_max :]

            # prepare
            rec_next = rec.clone()
            right_nodes = rec.gather(1, selected_index)
            argsort = rec.argsort()

            # new rec
            rec_next.scatter_(1, left, right)
            cur = left[:, :1].clone()
            for i in range(
                self.generator.num_loc - 2
            ):  # self.generator.num_loc - 2 is already correct
                next_cur = rec_next.gather(1, cur)
                pre_next_wrt_old = argsort.gather(1, next_cur)
                reverse_link_condition = (cur != pre_next_wrt_old) & ~(
                    (next_cur == right_nodes).any(-1, True)
                )
                next_next_cur = rec_next.gather(1, next_cur)
                rec_next.scatter_(
                    1,
                    next_cur,
                    torch.where(reverse_link_condition, pre_next_wrt_old, next_next_cur),
                )
                # if i >= self.generator.num_loc - 2: assert (reverse_link_condition == False).all()
                cur = next_cur

        return rec_next

    def _make_spec(self, generator: TSPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
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
                shape=(self.generator.num_loc),
                dtype=torch.int64,
            ),
            rec_best=Unbounded(
                shape=(self.generator.num_loc),
                dtype=torch.int64,
            ),
            visited_time=Unbounded(
                shape=(self.generator.num_loc, self.generator.num_loc),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(self.k_max * 3,),
            dtype=torch.int64,
            low=0,
            high=self.generator.num_loc,
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

    def get_mask(self, td):
        # return mask that is 1 if the corresponding action is feasible, 0 otherwise
        visited_time = td["visited_time"]
        bs, gs = visited_time.size()
        if self.two_opt_mode:
            selfmask = torch.eye(gs).view(1, gs, gs).to(td.device)
            masks = selfmask.expand(bs, gs, gs).bool()
            return ~masks
        else:
            assert False, "The masks for NeuOpt are handled within its policy"

    def _random_action(self, td):
        bs, gs = td["rec_best"].size()

        if self.two_opt_mode:
            mask = self.get_mask(td)
            logits = torch.rand(bs, gs, gs)
            logits[~mask] = -1e20
            prob = torch.softmax(logits.view(bs, -1), -1)
            sample = prob.multinomial(1)
            td["action"] = torch.cat((sample // (gs), sample % (gs)), -1)

        else:
            rec = td["rec_current"]
            visited_time = td["visited_time"]
            action_index = torch.zeros(bs, self.k_max, dtype=torch.long)
            k_action_left = torch.zeros(bs, self.k_max + 1, dtype=torch.long)
            k_action_right = torch.zeros(bs, self.k_max, dtype=torch.long)
            next_of_last_action = torch.zeros((bs, 1), dtype=torch.long) - 1
            mask = torch.zeros((bs, gs), dtype=torch.bool)
            stopped = torch.ones(bs, dtype=torch.bool)

            for i in range(self.k_max):
                # Sample action for a_i
                logits = torch.rand(bs, gs)
                logits[mask.clone()] = -1e30
                prob = torch.softmax(logits, -1)
                action = prob.multinomial(1)
                value_max, action_max = prob.max(-1, True)  ### fix bug of pytorch
                action = torch.where(
                    1 - value_max.view(-1, 1) < 1e-5, action_max.view(-1, 1), action
                )  ### fix bug of pytorch
                if i > 0:
                    action = torch.where(
                        stopped.unsqueeze(-1), action_index[:, :1], action
                    )

                # Store and Process actions
                next_of_new_action = rec.gather(1, action)
                action_index[:, i] = action.squeeze().clone()
                k_action_left[stopped, i] = action[stopped].squeeze().clone()
                k_action_right[~stopped, i - 1] = action[~stopped].squeeze().clone()
                k_action_left[:, i + 1] = next_of_new_action.squeeze().clone()

                # Process if k-opt close
                if i > 0:
                    stopped = stopped | (action == next_of_last_action).squeeze()
                else:
                    stopped = (action == next_of_last_action).squeeze()
                k_action_left[stopped, i] = k_action_left[stopped, i - 1]
                k_action_right[stopped, i] = k_action_right[stopped, i - 1]

                # Calc next basic masks
                if i == 0:
                    visited_time_tag = (
                        visited_time - visited_time.gather(1, action)
                    ) % gs
                mask &= False
                mask[(visited_time_tag <= visited_time_tag.gather(1, action))] = True
                if i == 0:
                    mask[visited_time_tag > (gs - 2)] = True
                mask[stopped, action[stopped].squeeze()] = (
                    False  # allow next k-opt starts immediately
                )
                # if True:#i == self.k_max - 2: # allow special case: close k-opt at the first selected node
                index_allow_first_node = (~stopped) & (
                    next_of_new_action.squeeze() == action_index[:, 0]
                )
                mask[index_allow_first_node, action_index[index_allow_first_node, 0]] = (
                    False
                )

                # Move to next
                next_of_last_action = next_of_new_action
                next_of_last_action[stopped] = -1

            # Form final action
            k_action_right[~stopped, -1] = k_action_left[~stopped, -1].clone()
            k_action_left = k_action_left[:, : self.k_max]
            td["action"] = torch.cat((action_index, k_action_left, k_action_right), -1)

        return td["action"]

    @classmethod
    def render(cls, td: TensorDict, actions: torch.Tensor = None, ax=None):
        solution_current = cls.get_current_solution(td)
        solution_best = cls.get_best_solution(td)
        return render_improvement(td, solution_current, solution_best)


class DenseRewardTSPEnv(TSPEnv):
    """
    This is an experimental version of the TSPEnv to be used with stepwise PPO. That is
    this environment defines a stepwise reward function for the TSP which is the distance added
    to the current tour by the given action.
    """

    def __init__(
        self, generator: TSPGenerator = None, generator_params: dict = {}, **kwargs
    ):
        super().__init__(
            generator,
            generator_params,
            check_solution=False,
            _torchrl_mode=True,
            **kwargs,
        )

    def _step(self, td):
        last_node = td["current_node"].clone()
        current_node = td["action"]

        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # calc stepwise reward
        last_node_loc = gather_by_index(td["locs"], last_node)
        curr_node_loc = gather_by_index(td["locs"], current_node)
        reward = get_distance(last_node_loc, curr_node_loc)[:, None]

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _get_reward(self, td, actions=None) -> TensorDict:
        if actions is not None:
            # Gather locations in order of tour and return distance between them (i.e., -reward)
            locs_ordered = gather_by_index(td["locs"], actions)
            return -get_tour_length(locs_ordered)
        return -td["reward"]
