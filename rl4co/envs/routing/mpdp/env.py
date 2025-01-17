from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .generator import MPDPGenerator
from .render import render

log = get_pylogger(__name__)


class MPDPEnv(RL4COEnvBase):
    """Multi-agent Pickup and Delivery Problem (mPDP) environment.
    The goal is to pick up and deliver all the packages while satisfying the precedence constraints.
    When an agent goes back to the depot, a new agent is spawned. In the min-max version, the goal is to minimize the
    maximum tour length among all agents. The reward is 0 unless the agent visits all the customers.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

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
        generator: MPDPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "mpdp"

    def __init__(
        self,
        generator: MPDPGenerator = None,
        generator_params: dict = {},
        objective: str = "minmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MPDPGenerator(**generator_params)
        self.generator = generator
        self.objective = objective
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        selected = td["action"][:, None]  # Add dimension for step

        agent_num = td["lengths"].size(1)
        n_loc = td["to_delivery"].size(-1) - agent_num - 1

        new_to_delivery = (selected + n_loc // 2) % (
            n_loc + agent_num + 1
        )  # the pair node of selected node

        is_request = (selected > agent_num) & (selected <= agent_num + n_loc // 2)
        td["left_request"][is_request] -= 1
        depot_distance = td["depot_distance"].scatter(-1, selected, 0)

        add_pd = td["add_pd_distance"][is_request.squeeze(-1), :].gather(
            -1, selected[is_request.squeeze(-1), :] - agent_num - 1
        )
        td["longest_lengths"][is_request.squeeze(-1), :].scatter_add_(
            -1, td["count_depot"][is_request.squeeze(-1), :], add_pd
        )
        td["add_pd_distance"][is_request.squeeze(-1), :].scatter_(
            -1, selected[is_request.squeeze(-1), :] - agent_num - 1, 0
        )
        remain_sum_paired_distance = td["add_pd_distance"].sum(-1, keepdim=True)
        remain_pickup_max_distance = depot_distance[:, : agent_num + 1 + n_loc // 2].max(
            dim=-1, keepdim=True
        )[0]
        remain_delivery_max_distance = depot_distance[
            :, agent_num + 1 + n_loc // 2 :
        ].max(dim=-1, keepdim=True)[0]

        # Calculate makespan
        cur_coord = gather_by_index(td["locs"], selected)
        path_lengths = (cur_coord - td["cur_coord"]).norm(p=2, dim=-1)

        td["lengths"].scatter_add_(-1, td["count_depot"], path_lengths.unsqueeze(-1))

        # If visit depot then plus one to count_depot\
        td["count_depot"][
            (selected == td["agent_idx"]) & (td["agent_idx"] < agent_num)
        ] += 1  # torch.ones(td["count_depot"][(selected == 0) & (td["agent_idx"] < agent_num)].shape, dtype=torch.int64, device=td["count_depot"].device)

        # `agent_idx` is added by 1 if the current agent comes back to depot
        agent_idx = (td["count_depot"] + 1) * torch.ones(
            selected.size(0), 1, dtype=torch.long, device=td["count_depot"].device
        )
        visited = td["visited"].scatter(-1, selected.unsqueeze(-1), 1)
        to_delivery = td["to_delivery"].scatter(-1, new_to_delivery[:, :, None], 1)

        # Get done and reward
        done = visited.all(dim=-1, keepdim=True).squeeze(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "visited": visited,
                "agent_idx": agent_idx,
                "cur_coord": cur_coord,
                "to_delivery": to_delivery,
                "depot_distance": depot_distance,
                "remain_sum_paired_distance": remain_sum_paired_distance,
                "remain_pickup_max_distance": remain_pickup_max_distance,
                "remain_delivery_max_distance": remain_delivery_max_distance,
                "i": td["i"] + 1,
                "done": done,
                "reward": reward,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
        agent_num: Optional[int] = None,  # NOTE hardcoded from ET
    ) -> TensorDict:
        device = td.device

        # NOTE: this is a hack to get the agent_num
        # agent_num = td["agent_num"][0].item() if agent_num is None else agent_num
        # agent_num = agent_num if agent_num is not None else td["agent_num"][0].item()

        depot = td["depot"]
        depot = depot.repeat(1, agent_num + 1, 1)
        loc = td["locs"]
        left_request = loc.size(1) // 2
        whole_instance = torch.cat((depot, loc), dim=1)

        # Distance from all nodes between each other
        distance = torch.cdist(whole_instance, whole_instance, p=2)
        index = torch.arange(left_request, 2 * left_request, device=device)[None, :, None]
        index = index.repeat(distance.shape[0], 1, 1)
        add_pd_distance = distance[
            :, agent_num + 1 : agent_num + 1 + left_request, agent_num + 1 :
        ].gather(-1, index)
        add_pd_distance = add_pd_distance.squeeze(-1)

        remain_pickup_max_distance = distance[:, 0, : agent_num + 1 + left_request].max(
            dim=-1, keepdim=True
        )[0]
        remain_delivery_max_distance = distance[:, 0, agent_num + 1 + left_request :].max(
            dim=-1, keepdim=True
        )[0]
        remain_sum_paired_distance = add_pd_distance.sum(dim=-1, keepdim=True)

        # Distance from depot to all nodes
        # Delivery nodes should consider the sum of distance from depot to paired pickup nodes and pickup nodes to delivery nodes
        distance[:, 0, agent_num + 1 : agent_num + 1 + left_request] = (
            distance[:, 0, agent_num + 1 : agent_num + 1 + left_request]
            + distance[:, 0, agent_num + 1 + left_request :]
        )

        # Distance from depot to all nodes
        depot_distance = distance[:, 0, :]
        depot_distance[:, agent_num + 1 : agent_num + 1 + left_request] = depot_distance[
            :, agent_num + 1 : agent_num + 1 + left_request
        ]  # + add_pd_distance

        batch_size, n_loc, _ = loc.size()
        to_delivery = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    n_loc // 2 + agent_num + 1,
                    dtype=torch.uint8,
                    device=device,
                ),
                torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=device),
            ],
            dim=-1,
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((depot, loc), -2),
                "visited": torch.zeros(
                    batch_size,
                    1,
                    n_loc + agent_num + 1,
                    dtype=torch.uint8,
                    device=device,
                ),
                "lengths": torch.zeros(batch_size, agent_num, device=device),
                "longest_lengths": torch.zeros(batch_size, agent_num, device=device),
                "cur_coord": (
                    td["depot"] if len(td["depot"].shape) == 2 else td["depot"].squeeze(1)
                ),
                "i": torch.zeros(
                    batch_size, dtype=torch.int64, device=device
                ),  # Vector with length num_steps
                "to_delivery": to_delivery,
                "count_depot": torch.zeros(
                    batch_size, 1, dtype=torch.int64, device=device
                ),
                "agent_idx": torch.ones(batch_size, 1, dtype=torch.long, device=device),
                "left_request": left_request
                * torch.ones(batch_size, 1, dtype=torch.long, device=device),
                "remain_pickup_max_distance": remain_pickup_max_distance,
                "remain_delivery_max_distance": remain_delivery_max_distance,
                "depot_distance": depot_distance,
                "remain_sum_paired_distance": remain_sum_paired_distance,
                "add_pd_distance": add_pd_distance,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Get the action mask for the current state."""

        visited_loc = td["visited"].clone()

        agent_num = td["lengths"].size(1)
        n_loc = visited_loc.size(-1) - agent_num - 1  # num of customers
        batch_size = visited_loc.size(0)
        agent_idx = td["agent_idx"][:, None, :]
        mask_loc = visited_loc.to(td["to_delivery"].device) | (1 - td["to_delivery"])

        # depot
        if td["i"][0].item() != 0:
            mask_loc[:, :, : agent_num + 1] = 1

            # if deliver nodes which is assigned agent is complete, then agent can go to depot
            no_item_to_delivery = (
                visited_loc[:, :, n_loc // 2 + agent_num + 1 :]
                == td["to_delivery"][:, :, n_loc // 2 + agent_num + 1 :]
            ).all(dim=-1)
            mask_loc[no_item_to_delivery.squeeze(-1), :, :] = mask_loc[
                no_item_to_delivery.squeeze(-1), :, :
            ].scatter_(-1, agent_idx[no_item_to_delivery.squeeze(-1), :, :], 0)

            condition = (td["count_depot"] == agent_num - 1) & (
                (visited_loc[:, :, agent_num + 1 :] == 0).sum(dim=-1) != 0
            )

            mask_loc[..., agent_num][condition] = 1

        else:
            return (
                torch.cat(
                    [
                        torch.zeros(
                            batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device
                        ),
                        torch.ones(
                            batch_size,
                            1,
                            n_loc + agent_num,
                            dtype=torch.uint8,
                            device=mask_loc.device,
                        ),
                    ],
                    dim=-1,
                )
                > 0
            )
        action_mask = mask_loc == 0  # action_mask gets feasible actions
        return action_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        # Calculate the reward (negative tour length)
        if self.objective == "minmax":
            return -td["lengths"].max(dim=-1, keepdim=True)[0].squeeze(-1)
        elif self.objective == "minsum":
            return -td["lengths"].sum(dim=-1, keepdim=True).squeeze(-1)
        else:
            raise ValueError(f"Unknown objective {self.objective}")

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        assert True, "Not implemented"

    def _make_spec(self, generator: MPDPGenerator):
        """Make the observation and action specs from the parameters."""
        max_nodes = self.num_loc + self.max_num_agents + 1
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(max_nodes, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(max_nodes, 1),
                dtype=torch.bool,
            ),
            visited=Unbounded(
                shape=(1, max_nodes),
                dtype=torch.bool,
            ),
            lengths=Unbounded(
                shape=(generator.max_num_agents,),
                dtype=torch.float32,
            ),
            longest_lengths=Unbounded(
                shape=(generator.max_num_agents,),
                dtype=torch.float32,
            ),
            cur_coord=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(2,),
                dtype=torch.float32,
            ),
            to_delivery=Unbounded(
                shape=(max_nodes, 1),
                dtype=torch.bool,
            ),
            count_depot=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            agent_idx=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            left_request=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            remain_pickup_max_distance=Unbounded(
                shape=(1,),
                dtype=torch.float32,
            ),
            remain_delivery_max_distance=Unbounded(
                shape=(1,),
                dtype=torch.float32,
            ),
            depot_distance=Unbounded(
                shape=(max_nodes,),
                dtype=torch.float32,
            ),
            remain_sum_paired_distance=Unbounded(
                shape=(1,),
                dtype=torch.float32,
            ),
            add_pd_distance=Unbounded(
                shape=(max_nodes,),
                dtype=torch.float32,
            ),
            ## NOTE: we should have a vectorized implementation for agent_num
            # agent_num=Unbounded(
            #     shape=(1,),
            #     dtype=torch.int64,
            # ),
            i=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=max_nodes,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
