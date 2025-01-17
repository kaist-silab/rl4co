import os
import sys

import pytest

from rl4co.envs import (
    ATSPEnv,
    FJSPEnv,
    JSSPEnv,
    PDPEnv,
    PDPRuinRepairEnv,
    TSPEnv,
    TSPkoptEnv,
)
from rl4co.models.rl import A2C, PPO, REINFORCE
from rl4co.models.zoo import (
    DACT,
    MDAM,
    N2S,
    POMO,
    ActiveSearch,
    AttentionModelPolicy,
    DeepACO,
    EASEmb,
    EASLay,
    HeterogeneousAttentionModel,
    L2DPPOModel,
    MatNet,
    NARGNNPolicy,
    NeuOpt,
    PolyNet,
    SymNCO,
)
from rl4co.utils import RL4COTrainer
from rl4co.utils.meta_trainer import ReptileCallback

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


def test_pomo_reptile():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )
    model = POMO(
        env,
        policy,
        batch_size=5,
        train_data_size=5 * 3,
        val_data_size=10,
        test_data_size=10,
    )
    meta_callback = ReptileCallback(
        data_type="size",
        sch_bar=0.9,
        num_tasks=2,
        alpha=0.99,
        alpha_decay=0.999,
        min_size=20,
        max_size=50,
    )
    trainer = RL4COTrainer(
        max_epochs=2,
        callbacks=[meta_callback],
        devices=1,
        accelerator=accelerator,
        limit_train_batches=3,
    )
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
@pytest.mark.skipfif("numba" not in sys.modules, reason="Numba not installed")
def test_deepaco():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = DeepACO(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        policy_kwargs={"n_ants": 5},
    )
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=1, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


def test_n2s():
    env = PDPRuinRepairEnv(generator_params=dict(num_loc=20))
    model = N2S(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_dact():
    env = TSPkoptEnv(generator_params=dict(num_loc=20), k_max=2)
    model = DACT(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
        CL_best=True,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_neuopt():
    env = TSPkoptEnv(generator_params=dict(num_loc=20), k_max=4)
    model = NeuOpt(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
        CL_best=True,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.parametrize("env_cls", [FJSPEnv, JSSPEnv])
def test_l2d_ppo(env_cls):
    env = env_cls(stepwise_reward=True, _torchrl_mode=True)
    model = L2DPPOModel(
        env, train_data_size=10, val_data_size=10, test_data_size=10, buffer_size=1000
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_polynet():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = PolyNet(
        env,
        k=10,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)
