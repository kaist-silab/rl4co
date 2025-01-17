from functools import lru_cache, cached_property, partial
from typing import Optional, Tuple

import numpy as np
import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from rl4co.utils.decoding import Sampling
from rl4co.utils.ops import get_distance_matrix, unbatchify


class AntSystem:
    """Implements the Ant System algorithm: https://doi.org/10.1109/3477.484436.

    Args:
        log_heuristic: Logarithm of the heuristic matrix.
        n_ants: Number of ants to be used in the algorithm. Defaults to 20.
        alpha: Importance of pheromone in the decision-making process. Defaults to 1.0.
        beta: Importance of heuristic information in the decision-making process. Defaults to 1.0.
        decay: Rate at which pheromone evaporates. Should be between 0 and 1. Defaults to 0.95.
        Q: Rate at which pheromone deposits. Defaults to `1 / n_ants`.
        temperature: Temperature for the softmax during decoding. Defaults to 0.1.
        pheromone: Initial pheromone matrix. Defaults to `torch.ones_like(log_heuristic)`.
        require_logprobs: Whether to require the log probability of actions. Defaults to False.
        use_local_search: Whether to use local_search provided by the env. Default to False.
        use_nls: Whether to use neural-guided local search provided by the env. Default to False.
        n_perturbations: Number of perturbations to be used for nls. Defaults to 5.
        local_search_params: Arguments to be passed to the local_search.
        perturbation_params: Arguments to be passed to the perturbation used for nls.
    """

    def __init__(
        self,
        log_heuristic: Tensor,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.95,
        Q: Optional[float] = None,
        temperature: float = 0.1,
        pheromone: Optional[Tensor] = None,
        require_logprobs: bool = False,
        use_local_search: bool = False,
        use_nls: bool = False,
        n_perturbations: int = 5,
        local_search_params: dict = {},
        perturbation_params: dict = {},
        start_node: Optional[int] = None,
    ):
        self.batch_size = log_heuristic.shape[0]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.Q = 1 / self.n_ants if Q is None else Q
        self.temperature = temperature

        self.log_heuristic = log_heuristic / self.temperature

        if pheromone is None:
            self.pheromone = torch.ones_like(log_heuristic)
            self.pheromone.fill_(0.0005)
        else:
            self.pheromone = pheromone

        self.final_actions = self.final_reward = None
        self.require_logprobs = require_logprobs
        self.all_records = []

        self.use_local_search = use_local_search
        assert not (use_nls and not use_local_search), "use_nls requires use_local_search"
        self.use_nls = use_nls
        self.n_perturbations = n_perturbations
        self.local_search_params = local_search_params
        self.perturbation_params = perturbation_params
        self.start_node = start_node

        self._batchindex = torch.arange(self.batch_size, device=log_heuristic.device)

    @cached_property
    def heuristic_dist(self) -> torch.Tensor:
        heuristic = self.log_heuristic.exp().detach().cpu() + 1e-10
        heuristic_dist = 1 / (heuristic / heuristic.max(-1, keepdim=True)[0] + 1e-5)
        heuristic_dist[:, torch.arange(heuristic_dist.shape[1]), torch.arange(heuristic_dist.shape[2])] = 0
        return heuristic_dist

    @staticmethod
    def select_start_node_fn(
        td: TensorDict, env: RL4COEnvBase, num_starts: int, start_node: Optional[int]=None
    ):
        if env.name == "tsp" and start_node is not None:
            # For now, only TSP supports explicitly setting the start node
            return start_node * torch.ones(
                td.shape[0] * num_starts, dtype=torch.long, device=td.device
            )

        # if start_node is not set, we use random start nodes
        return torch.multinomial(td["action_mask"].float(), num_starts, replacement=True).view(-1)

    def run(
        self,
        td_initial: TensorDict,
        env: RL4COEnvBase,
        n_iterations: int,
    ) -> Tuple[TensorDict, Tensor, Tensor]:
        """Run the Ant System algorithm for a specified number of iterations.

        Args:
            td_initial: Initial state of the problem.
            env: Environment representing the problem.
            n_iterations: Number of iterations to run the algorithm.

        Returns:
            td: The final state of the problem.
            actions: The final actions chosen by the algorithm.
            reward: The final reward achieved by the algorithm.
        """
        for _ in range(n_iterations):
            # reset environment
            td = td_initial.clone()
            self._one_step(td, env)

        action_matrix = self._convert_final_action_to_matrix()
        assert action_matrix is not None and self.final_reward is not None
        td, env = self._recreate_final_routes(td_initial, env, action_matrix)

        return td, action_matrix, self.final_reward

    def _one_step(self, td: TensorDict, env: RL4COEnvBase):
        """Run one step of the Ant System algorithm.

        Args:
            td: Current state of the problem.
            env: Environment representing the problem.

        Returns:
            actions: The actions chosen by the algorithm.
            reward: The reward achieved by the algorithm.
        """
        # sampling
        td, env, actions, reward = self._sampling(td, env)
        # local search, reserved for extensions
        if self.use_local_search:
            actions, reward = self.local_search(td, env, actions)

        # reshape from (batch_size * n_ants, ...) to (batch_size, n_ants, ...)
        reward = unbatchify(reward, self.n_ants)
        actions = unbatchify(actions, self.n_ants)

        # update final actions and rewards
        self._update_results(actions, reward)
        # update pheromone matrix
        self._update_pheromone(actions, reward)

        return actions, reward

    def _sampling(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
    ):
        # Sample from heatmaps
        # p = phe**alpha * heu**beta <==> log(p) = alpha*log(phe) + beta*log(heu)
        heatmaps_logits = (
            self.alpha * torch.log(self.pheromone) + self.beta * self.log_heuristic
        )
        decode_strategy = Sampling(
            multistart=True,
            num_starts=self.n_ants,
            select_start_nodes_fn=partial(self.select_start_node_fn, start_node=self.start_node),
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        while not td["done"].all():
            logits, mask = NonAutoregressiveDecoder.heatmap_to_logits(
                td, heatmaps_logits, num_starts
            )
            td = decode_strategy.step(logits, mask, td)
            td = env.step(td)["next"]

        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
        reward = env.get_reward(td, actions)

        if self.require_logprobs:
            self.all_records.append((logprobs, actions, reward, td.get("mask", None)))

        return td, env, actions, reward

    def local_search(
        self, td: TensorDict, env: RL4COEnvBase, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Perform local search on the actions and reward obtained.

        Args:
            td: Current state of the problem.
            env: Environment representing the problem.
            actions: Actions chosen by the algorithm.

        Returns:
            actions: The modified actions
            reward: The modified reward
        """
        td_cpu = td.detach().cpu()  # Convert to CPU in advance to minimize the overhead from device transfer
        td_cpu["distances"] = get_distance_matrix(td_cpu["locs"])
        # TODO: avoid or generalize this, e.g., pre-compute for local search in each env
        actions = actions.detach().cpu()
        best_actions = env.local_search(td=td_cpu, actions=actions, **self.local_search_params)
        best_rewards = env.get_reward(td_cpu, best_actions)

        if self.use_nls:
            td_cpu_perturb = td_cpu.clone()
            td_cpu_perturb["distances"] = torch.tile(self.heuristic_dist, (self.n_ants, 1, 1))
            new_actions = best_actions.clone()

            for _ in range(self.n_perturbations):
                perturbed_actions = env.local_search(
                    td=td_cpu_perturb, actions=new_actions, **self.perturbation_params
                )
                new_actions = env.local_search(td=td_cpu, actions=perturbed_actions, **self.local_search_params)
                new_rewards = env.get_reward(td_cpu, new_actions)

                improved_indices = new_rewards > best_rewards
                best_actions = env.replace_selected_actions(best_actions, new_actions, improved_indices)
                best_rewards[improved_indices] = new_rewards[improved_indices]

        best_actions = best_actions.to(td.device)
        best_rewards = best_rewards.to(td.device)

        return best_actions, best_rewards

    def _update_results(self, actions, reward):
        # update the best-trails recorded in self.final_actions
        best_index = reward.argmax(-1)
        best_reward = reward[self._batchindex, best_index]
        best_actions = actions[self._batchindex, best_index]

        if self.final_actions is None or self.final_reward is None:
            self.final_actions = list(iter(best_actions.clone()))
            self.final_reward = best_reward.clone()
        else:
            require_update = self._batchindex[self.final_reward <= best_reward]
            for index in require_update:
                self.final_actions[index] = best_actions[index]
            self.final_reward[require_update] = best_reward[require_update]

        return best_index

    def _update_pheromone(self, actions, reward):
        # calculate Δphe
        delta_pheromone = torch.zeros_like(self.pheromone)
        from_node = actions
        to_node = torch.roll(from_node, -1, -1)
        mapped_reward = self._reward_map(reward).detach()
        batch_action_indices = self._batch_action_indices(
            self.batch_size, actions.shape[-1], reward.device
        )

        for ant_index in range(self.n_ants):
            delta_pheromone[
                batch_action_indices,
                from_node[:, ant_index].flatten(),
                to_node[:, ant_index].flatten(),
            ] += mapped_reward[batch_action_indices, ant_index]

        # decay & update
        self.pheromone *= self.decay
        self.pheromone += delta_pheromone

    def _reward_map(self, x: Tensor):
        """Map reward $f: \\mathbb{R} \\rightarrow \\mathbb{R}^+$"""
        M, _ = x.max(-1, keepdim=True)
        m, _ = x.min(-1, keepdim=True)
        v = ((x - m) / (M - m)) ** 2 * self.Q
        return v

    def _recreate_final_routes(self, td, env, action_matrix):
        for action_index in range(action_matrix.shape[-1]):
            actions = action_matrix[:, action_index]
            td.set("action", actions)
            td = env.step(td)["next"]

        assert td["done"].all()
        return td, env

    def get_logp(self):
        """Get the log probability (logprobs) values recorded during the execution of the algorithm.

        Returns:
            results: Tuple containing the log probability values,
                actions chosen, rewards obtained, and mask values (if available).

        Raises:
            AssertionError: If `require_logp` is not enabled.
        """

        assert (
            self.require_logprobs
        ), "Please enable `require_logp` to record logprobs values"

        logprobs_list, actions_list, reward_list, mask_list = [], [], [], []

        for logprobs, actions, reward, mask in self.all_records:
            logprobs_list.append(logprobs)
            actions_list.append(actions)
            reward_list.append(reward)
            mask_list.append(mask)

        if mask_list[0] is None:
            mask_list = None
        else:
            mask_list = torch.stack(mask_list, 0)

        # reset records
        self.all_records = []

        return (
            torch.stack(logprobs_list, 0),
            torch.stack(actions_list, 0),
            torch.stack(reward_list, 0),
            mask_list,
        )

    @staticmethod
    @lru_cache(5)
    def _batch_action_indices(batch_size: int, n_actions: int, device: torch.device):
        batchindex = torch.arange(batch_size, device=device)
        return batchindex.unsqueeze(1).repeat(1, n_actions).view(-1)

    def _convert_final_action_to_matrix(self) -> Optional[Tensor]:
        if self.final_actions is None:
            return None
        action_count = max(len(actions) for actions in self.final_actions)
        mat_actions = torch.zeros(
            (self.batch_size, action_count),
            device=self.final_actions[0].device,
            dtype=self.final_actions[0].dtype,
        )
        for index, action in enumerate(self.final_actions):
            mat_actions[index, : len(action)] = action

        return mat_actions
