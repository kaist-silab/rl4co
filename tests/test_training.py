import os
import sys

import pytest

from rl4co.envs import ATSPEnv, PDPEnv, TSPEnv
from rl4co.models.rl import A2C, PPO, REINFORCE
from rl4co.models.zoo import (
    MDAM,
    ActiveSearch,
    AttentionModelPolicy,
    DeepACO,
    EASEmb,
    EASLay,
    HeterogeneousAttentionModel,
    MatNet,
    NARGNNPolicy,
    SymNCO,
    GLOP,
)
from rl4co.utils import RL4COTrainer

# Get env variable MAC_OS_GITHUB_RUNNER
if "MAC_OS_GITHUB_RUNNER" in os.environ:
    accelerator = "cpu"
else:
    accelerator = "auto"


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "mean", "no", "critic"])
def test_reinforce(baseline):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = REINFORCE(
        env,
        policy,
        baseline=baseline,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_a2c():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = A2C(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_ppo():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = PPO(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=None, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


def test_symnco():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = SymNCO(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        num_augment=2,
        num_starts=20,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_ham():
    env = PDPEnv(generator_params=dict(num_loc=20))
    model = HeterogeneousAttentionModel(
        env, train_data_size=10, val_data_size=10, test_data_size=10
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_matnet():
    env = ATSPEnv(generator_params=dict(num_loc=20))
    model = MatNet(
        env,
        baseline="shared",
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_mdam():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = MDAM(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.parametrize("SearchMethod", [ActiveSearch, EASEmb, EASLay])
def test_search_methods(SearchMethod):
    env = TSPEnv(generator_params=dict(num_loc=20))
    batch_size = 2 if SearchMethod not in [ActiveSearch] else 1
    dataset = env.dataset(2)
    policy = AttentionModelPolicy(env_name=env.name)
    model = SearchMethod(env, policy, dataset, max_iters=2, batch_size=batch_size)
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(
    "torch_geometric" not in sys.modules, reason="PyTorch Geometric not installed"
)
def test_nargnn():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = NARGNNPolicy(env_name=env.name)
    model = REINFORCE(
        env, policy=policy, train_data_size=10, val_data_size=10, test_data_size=10
    )
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=None, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(
    "torch_geometric" not in sys.modules, reason="PyTorch Geometric not installed"
)
def test_deepaco():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = DeepACO(env, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=1, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)
