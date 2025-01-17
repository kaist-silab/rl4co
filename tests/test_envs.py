import warnings

import matplotlib.pyplot as plt
import pytest
import torch

from tensordict import TensorDict

from rl4co.envs import (
    ATSPEnv,
    CVRPEnv,
    CVRPTWEnv,
    DPPEnv,
    FFSPEnv,
    FJSPEnv,
    JSSPEnv,
    MDCPDPEnv,
    MDPPEnv,
    MTSPEnv,
    MTVRPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SMTWTPEnv,
    SPCTSPEnv,
    SVRPEnv,
    TSPEnv,
    FLPEnv,
    MCPEnv,
)
from rl4co.utils.decoding import random_policy, rollout

# Switch to non-GUI backend for testing
plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


@pytest.mark.parametrize(
    "env_cls",
    [
        TSPEnv,
        CVRPEnv,
        CVRPTWEnv,
        SVRPEnv,
        SDVRPEnv,
        PCTSPEnv,
        SPCTSPEnv,
        OPEnv,
        PDPEnv,
        MTSPEnv,
        ATSPEnv,
        MDCPDPEnv,
    ],
)
def test_routing(env_cls, batch_size=2, size=20):
    env = env_cls(generator_params=dict(num_loc=size))
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize(
    "variant",
    [
        "all",
        "cvrp",
        "ovrp",
        "vrpb",
        "vrpl",
        "vrptw",
        "ovrptw",
        "ovrpb",
        "ovrpl",
        "vrpbl",
        "vrpbtw",
        "vrpltw",
        "ovrpbl",
        "ovrpltw",
        "vrpltw",
        "ovrpbltw",
    ],
)
def test_mtvrp(variant, batch_size=2, size=20):
    env = MTVRPEnv(generator_params=dict(num_loc=size, variant_preset=variant))
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [DPPEnv, MDPPEnv])
def test_eda(env_cls, batch_size=2, max_decaps=5):
    env = env_cls(max_decaps=max_decaps)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [FFSPEnv, FJSPEnv, JSSPEnv])
@pytest.mark.parametrize("mask_no_ops", [True, False])
def test_scheduling(env_cls, mask_no_ops, batch_size=2):
    env = env_cls()
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [SMTWTPEnv])
def test_smtwtp(env_cls, batch_size=2):
    env = env_cls(num_job=4)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [JSSPEnv])
def test_jssp_lb(env_cls):
    env = env_cls(generator_params={"num_jobs": 2, "num_machines": 2})
    td = TensorDict(
        {
            "proc_times": torch.tensor(
                [[[1, 0, 0, 4], [0, 2, 3, 0]]], dtype=torch.float32
            ),
            "start_op_per_job": torch.tensor([[0, 2]], dtype=torch.long),
            "end_op_per_job": torch.tensor([[1, 3]], dtype=torch.long),
            "pad_mask": torch.tensor([[0, 0, 0, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )

    td = env.reset(td)

    actions = [0, 1, 1]
    for action in actions:
        # NOTE add 1 to account for dummy action (waiting)
        td.set("action", torch.tensor([action + 1], dtype=torch.long))
        td = env.step(td)["next"]

    lb_expected = torch.tensor([[1, 5, 3, 7]], dtype=torch.float32)
    assert torch.allclose(td["lbs"], lb_expected)


@pytest.mark.parametrize("env_cls", [FLPEnv, MCPEnv])
def test_flp_mcp(env_cls, batch_size=2):
    env = env_cls()
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


def test_scheduling_dataloader():
    from tempfile import TemporaryDirectory

    from rl4co.envs.scheduling.fjsp.parser import write

    write_env = FJSPEnv()

    td = write_env.reset(batch_size=[2])
    with TemporaryDirectory() as tmpdirname:
        write(tmpdirname, td)
        read_env = FJSPEnv(generator_params={"file_path": tmpdirname})
        td = read_env.reset(batch_size=2)
    assert td.size(0) == 2
