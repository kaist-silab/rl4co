import os

from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import DPPGenerator

log = get_pylogger(__name__)


class DPPEnv(RL4COEnvBase):
    """Decap Placement Problem (DPP) as done in DevFormer paper: https://arxiv.org/abs/2205.13225

    The environment is a 10x10 grid with 100 locations containing either a probing port or a keepout region.
    The goal is to place decaps (decoupling capacitors) to maximize the impedance suppression at the probing port.
    Decaps cannot be placed in keepout regions or at the probing port and the number of decaps is limited.

    Observations:
        - locations of the probing port and keepout regions
        - current decap placement
        - remaining decaps

    Constraints:
        - decaps cannot be placed at the probing port or keepout regions
        - the number of decaps is limited

    Finish Condition:
        - the number of decaps exceeds the limit

    Reward:
        - the impedance suppression at the probing port

    Args:
        generator: DPPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "dpp"

    def __init__(
        self,
        generator: DPPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = DPPGenerator(**generator_params)
        self.generator = generator

        self.max_decaps = self.generator.max_decaps
        self.size = self.generator.size
        self.raw_pdn = self.generator.raw_pdn
        self.decap = self.generator.decap
        self.freq = self.generator.freq
        self.num_freq = self.generator.num_freq
        self.data_dir = self.generator.data_dir

        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # Set available to 0 (i.e., already placed) if the current node is the first node
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # Set done if i is greater than max_decaps
        done = td["i"] >= self.max_decaps - 1

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        # Other variables
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": td["locs"],
                "probe": td["probe"],
                "i": i,
                "action_mask": td["action_mask"],
                "keepout": ~td["action_mask"],
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: DPPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.size**2, 2),
                dtype=torch.float32,
            ),
            probe=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            keepout=Unbounded(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.size**2,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td, actions):
        """
        We call the reward function with the final sequence of actions to get the reward
        Calling per-step would be very time consuming due to decap simulation
        """
        # We do the operation in a batch
        if len(td.batch_size) == 0:
            td = td.unsqueeze(0)
            actions = actions.unsqueeze(0)
        probes = td["probe"]
        reward = torch.stack(
            [self._decap_simulator(p, a) for p, a in zip(probes, actions)]
        )
        return reward

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert True, "Not implemented"

    def _decap_placement(self, pi, probe):
        device = pi.device

        n = m = self.size  # columns and rows
        num_decap = torch.numel(pi)
        z1 = self.raw_pdn.to(device)

        decap = self.decap.reshape(-1).to(device)
        z2 = torch.zeros(
            (self.num_freq, num_decap, num_decap), dtype=torch.float32, device=device
        )

        qIndx = torch.arange(num_decap, device=device)

        z2[:, qIndx, qIndx] = torch.abs(decap)[:, None].repeat_interleave(
            z2[:, qIndx, qIndx].shape[-1], dim=-1
        )
        pIndx = pi.long()

        aIndx = torch.arange(len(z1[0]), device=device)
        aIndx = torch.tensor(
            list(set(aIndx.tolist()) - set(pIndx.tolist())), device=device
        )

        z1aa = z1[:, aIndx, :][:, :, aIndx]
        z1ap = z1[:, aIndx, :][:, :, pIndx]
        z1pa = z1[:, pIndx, :][:, :, aIndx]
        z1pp = z1[:, pIndx, :][:, :, pIndx]
        z2qq = z2[:, qIndx, :][:, :, qIndx]

        zout = z1aa - torch.matmul(torch.matmul(z1ap, torch.inverse(z1pp + z2qq)), z1pa)

        idx = torch.arange(n * m, device=device)
        mask = torch.zeros(n * m, device=device).bool()
        mask[pi] = True
        mask = mask & (idx < probe)
        probe -= mask.sum().item()

        zout = zout[:, probe, probe]
        return zout

    def _decap_model(self, z_initial, z_final):
        impedance_gap = torch.zeros(self.num_freq, device=self.device)

        impedance_gap = z_initial - z_final
        reward = torch.sum(impedance_gap * 1000000000 / self.freq.to(self.device))

        reward = reward / 10
        return reward

    def _initial_impedance(self, probe):
        zout = self.raw_pdn.to(self.device)[:, probe, probe]
        return zout

    def _decap_simulator(self, probe, solution, keepout=None):
        self.to(self.device)

        probe = probe.item()

        assert len(solution) == len(
            torch.unique(solution)
        ), "An Element of Decap Sequence must be Unique"

        if keepout is not None:
            keepout = torch.tensor(keepout)
            intersect = torch.tensor(list(set(solution.tolist()) & set(keepout.tolist())))
            assert len(intersect) == 0, "Decap must be not placed at the keepout region"

        z_initial = self._initial_impedance(probe)
        z_initial = torch.abs(z_initial)
        z_final = self._decap_placement(solution, probe)
        z_final = torch.abs(z_final)
        reward = self._decap_model(z_initial, z_final)
        return reward

    def _load_dpp_data(self, chip_file, decap_file, freq_file):
        def _load_file(fpath):
            f = os.path.join(self.generator.data_dir, fpath)
            if not os.path.isfile(f):
                self._download_data()
            with open(f, "rb") as f_:
                return torch.from_numpy(np.load(f_)).to(self.device)

        self.raw_pdn = _load_file(chip_file)  # [num_freq, size^2, size^2]
        self.decap = _load_file(decap_file).to(torch.complex64)  # [num_freq, 1, 1]
        self.freq = _load_file(freq_file)  # [num_freq]
        self.size = int(np.sqrt(self.raw_pdn.shape[-1]))
        self.num_freq = self.freq.shape[0]
