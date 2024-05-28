import warnings

import matplotlib.pyplot as plt
import pytest
import torch

from rl4co.envs import (
    ATSPEnv,
    CVRPEnv,
    CVRPTWEnv,
    DPPEnv,
    FFSPEnv,
    FJSPEnv,
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
    SHPPEnv,
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
        FJSPEnv,
        SHPPEnv,
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


@pytest.mark.parametrize("env_cls", [FFSPEnv])
def test_scheduling(env_cls, batch_size=2):
    env = env_cls()
    td = env.reset(batch_size=[batch_size])
    td["action"] = torch.tensor([1, 1])
    td = env._step(td)


@pytest.mark.parametrize("env_cls", [SMTWTPEnv])
def test_smtwtp(env_cls, batch_size=2):
    env = env_cls(num_job=4)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)
