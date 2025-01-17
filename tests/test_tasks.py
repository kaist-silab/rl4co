import pyrootutils
import pytest

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from rl4co.envs import TSPEnv
from rl4co.models import AttentionModelPolicy
from rl4co.tasks.eval import evaluate_policy
from rl4co.tasks.train import run


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name="main.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".gitignore"))
            cfg.trainer.max_epochs = 1
            cfg.model.train_data_size = 100
            cfg.model.val_data_size = 100
            cfg.model.test_data_size = 100
            cfg.model.batch_size = 2  # faster for CPU (not sure exactly why)
            cfg.env.val_file = None  # validate on self-generated data
            cfg.env.test_file = None  # validate on self-generated data
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.callbacks.learning_rate_monitor = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


def test_train_fast_dev_run(cfg_train):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    run(cfg_train)


@pytest.mark.parametrize(
    "method",
    ["greedy", "sampling", "multistart_greedy", "augment", "multistart_greedy_augment"],
)
def test_eval(method):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    out = evaluate_policy(env, policy, env.dataset(3), method=method)
    assert out["rewards"].shape == (3,)
