from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger

from .generator import SVRPGenerator

log = get_pylogger(__name__)


class SVRPEnv(RL4COEnvBase):
    """Skill-Vehicle Routing Problem (SVRP) environment.
    Basic Skill-VRP environment. The environment is a variant of the Capacitated Vehicle Routing Problem (CVRP).
    Each technician has a certain skill-level and each customer node requires a certain skill-level to be serviced.
    Each customer node needs is to be serviced by exactly one technician. Technicians can only service nodes if
    their skill-level is greater or equal to the required skill-level of the node. The environment is episodic and
    the goal is to minimize the total travel cost of the technicians. The travel cost depends on the skill-level of
    the technician. The environment is defined by the following parameters:

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
    """

    name = "svrp"

    def __init__(
        self,
        generator: SVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SVRPGenerator(**generator_params)
        self.generator = generator
        self.tech_costs = self.generator.tech_costs
        self._make_spec(self.generator)

    def _make_spec(self, generator):
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
            skills=Bounded(
                low=generator.min_skill,
                high=generator.max_skill,
                shape=(generator.num_loc, 1),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
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
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Calculates the action mask for the Skill-VRP. The action mask is a binary mask that indicates which customer nodes can be services, given the previous decisions.
        For the Skill-VRP, a node can be serviced if the technician has the required skill-level and the node has not been visited yet.
        The depot cannot be visited if there are still unserved nodes and the technician either just visited the depot or is the last technician
        (because every customer node needs to be visited).
        """
        batch_size = td["locs"].shape[0]
        # check skill level
        current_tech_skill = gather_by_index(td["techs"], td["current_tech"]).reshape(
            [batch_size, 1]
        )
        can_service = td["skills"] <= current_tech_skill.unsqueeze(1).expand_as(
            td["skills"]
        )
        mask_loc = td["visited"][..., 1:, :].to(can_service.dtype) | ~can_service
        # Cannot visit the depot if there are still unserved nodes and I either just visited the depot or am the last technician
        mask_depot = (
            (td["current_node"] == 0) | (td["current_tech"] == td["techs"].size(-2) - 1)
        ) & ((mask_loc == 0).int().sum(-2) > 0)
        return ~torch.cat((mask_depot[..., None], mask_loc), -2).squeeze(-1)

    def _step(self, td: TensorDict) -> torch.Tensor:
        """Step function for the Skill-VRP. If a technician returns to the depot, the next technician is sent out.
        The visited node is marked as visited. The reward is set to zero and the done flag is set if all nodes have been visited.
        """
        current_node = td["action"][:, None]  # Add dimension for step

        # if I go back to the depot, send out next technician
        td["current_tech"] += (current_node == 0).int()

        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-2, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-2) == visited.size(-2)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "techs": td["techs"],
                "skills": td["skills"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_tech": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1, 1),
                    dtype=torch.uint8,
                    device=device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Calculated the reward, where the reward is the negative total travel cost of the technicians.
        The travel cost depends on the skill-level of the technician."""
        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather dataset in order of tour
        batch_size = td["locs"].shape[0]
        depot = td["locs"][..., 0:1, :]
        locs_ordered = torch.cat(
            [
                depot,
                gather_by_index(td["locs"], actions).reshape(
                    [batch_size, actions.size(-1), 2]
                ),
            ],
            dim=1,
        )

        # calculate travelling costs depending on the technicians' skill level
        costs = torch.zeros(batch_size, locs_ordered.size(-2), device=self.device)
        indices = torch.nonzero(actions == 0)
        start = tech = 0
        batch = 0
        for each in indices:
            if each[0] > batch:
                costs[batch, start:] = self.tech_costs[tech]
                start = tech = 0
                batch = each[0]
            end = (
                each[-1] + 1
            )  # indices in locs_ordered are shifted by one due to added depot in the front
            costs[batch, start:end] = self.tech_costs[tech]
            tech += 1
            start = end
        costs[batch, start:] = self.tech_costs[tech]

        travel_to = torch.roll(locs_ordered, -1, dims=-2)
        distances = get_distance(locs_ordered, travel_to)
        return -(distances * costs).sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are not visited twice except depot and required skill levels are always met."""
        batch_size, graph_size = td["skills"].shape[0], td["skills"].shape[1]
        sorted_pi = actions.data.sort(1).values

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # make sure all required skill  levels are met
        indices = torch.nonzero(actions == 0)
        skills = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=td.device), td["skills"]], 1
        )
        skills_ordered = gather_by_index(skills, actions).reshape(
            [batch_size, actions.size(-1), 1]
        )
        batch = start = tech = 0
        for each in indices:
            if each[0] > batch:
                start = tech = 0
                batch = each[0]
            assert (
                skills_ordered[batch, start : each[1]] <= td["techs"][batch, tech]
            ).all(), "Skill level not met"
            start = each[1] + 1  # skip the depot
            tech += 1
