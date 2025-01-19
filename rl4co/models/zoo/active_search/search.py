import time

from functools import partial
from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset

from rl4co.data.transforms import StateAugmentation
from rl4co.models.common.transductive import TransductiveModel
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ActiveSearch(TransductiveModel):
    """Active Search for Neural Combination Optimization from Bello et al. (2016).
    Fine-tunes the whole policy network (encoder + decoder) on a batch of instances.
    Reference: https://arxiv.org/abs/1611.09940

    Args:
        env: RL4CO environment to be solved
        policy: policy network
        dataset: dataset to be used for training
        batch_size: batch size for training
        max_iters: maximum number of iterations
        augment_size: number of augmentations per state
        augment_dihedral: whether to augment with dihedral rotations
        parallel_runs: number of parallel runs
        max_runtime: maximum runtime in seconds
        save_path: path to save solution checkpoints
        optimizer: optimizer to use for training
        optimizer_kwargs: keyword arguments for optimizer
        **kwargs: additional keyword arguments
    """

    def __init__(
        self,
        env,
        policy,
        dataset: Dataset | str,
        batch_size: int = 1,
        max_iters: int = 200,
        augment_size: int = 8,
        augment_dihedral: bool = True,
        num_parallel_runs: int = 1,
        max_runtime: int = 86_400,
        save_path: str = None,
        optimizer: str | torch.optim.Optimizer | partial = "Adam",
        optimizer_kwargs: dict = {"lr": 2.6e-4, "weight_decay": 1e-6},
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        assert batch_size == 1, "Batch size must be 1 for active search"

        super(ActiveSearch, self).__init__(
            env,
            policy=policy,
            dataset=dataset,
            batch_size=batch_size,
            max_iters=max_iters,
            max_runtime=max_runtime,
            save_path=save_path,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def setup(self, stage="fit"):
        """Setup base class and instantiate:
        - augmentation
        - instance solutions and rewards
        - original policy state dict
        """
        log.info("Setting up active search...")
        super(ActiveSearch, self).setup(stage)

        # Instantiate augmentation
        self.augmentation = StateAugmentation(
            num_augment=self.hparams.augment_size,
            augment_fn="dihedral8" if self.hparams.augment_dihedral else "symmetric",
        )

        # Store original policy state dict
        self.original_policy_state = self.policy.state_dict()

        # Get dataset size and problem size
        dataset_size = len(self.dataset)
        _batch = next(iter(self.train_dataloader()))
        self.problem_size = self.env.reset(_batch)["action_mask"].shape[-1]
        self.instance_solutions = torch.zeros(
            dataset_size, self.problem_size * 2, dtype=int
        )
        self.instance_rewards = torch.zeros(dataset_size)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Called before training (i.e. search) for a new batch begins.
        We re-load the original policy state dict and configure the optimizer.
        """
        self.policy.load_state_dict(self.original_policy_state)
        self.configure_optimizers(self.policy.parameters())

    def training_step(self, batch, batch_idx):
        """Main search loop. We use the training step to effectively adapt to a `batch` of instances."""
        # Augment state
        batch_size = batch.shape[0]
        td_init = self.env.reset(batch)
        n_aug, n_start, n_runs = (
            self.augmentation.num_augment,
            self.env.get_num_starts(td_init),
            self.hparams.num_parallel_runs,
        )
        td_init = self.augmentation(td_init)
        td_init = batchify(td_init, n_runs)

        # Solution and reward buffer
        max_reward = torch.full((batch_size,), -float("inf"), device=batch.device)
        best_solutions = torch.zeros(
            batch_size, self.problem_size * 2, device=batch.device, dtype=int
        )

        # Init search
        t_start = time.time()
        for i in range(self.hparams.max_iters):
            # Evaluate policy with sampling multistarts (as in POMO)
            out = self.policy(
                td_init.clone(),
                env=self.env,
                decode_type="multistart_sampling",
                num_starts=n_start,
            )

            if i == 0:
                log.info(f"Initial reward: {out['reward'].max():.2f}")

            # Update best solution and reward found
            max_reward_iter = out["reward"].max()
            if max_reward_iter > max_reward:
                max_reward_idx = out["reward"].argmax()
                best_solution_iter = out["actions"][max_reward_idx]
                max_reward = max_reward_iter
                best_solutions[0, : best_solution_iter.shape[0]] = best_solution_iter

            # Compute REINFORCE loss with shared baseline
            reward = unbatchify(out["reward"], (n_runs, n_aug, n_start))
            ll = unbatchify(out["log_likelihood"], (n_runs, n_aug, n_start))
            advantage = reward - reward.mean(dim=-1, keepdim=True)
            loss = -(advantage * ll).mean()

            # Backpropagate loss
            # perform manual optimization following the Lightning routine
            # https://lightning.ai/docs/pytorch/stable/common/optimization.html
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)

            self.log_dict(
                {
                    "loss": loss,
                    "max_reward": max_reward,
                    "step": i,
                    "time": time.time() - t_start,
                },
                on_step=self.log_on_step,
            )

            # Stop if max runtime is exceeded
            if time.time() - t_start > self.hparams.max_runtime:
                break

        return {"max_reward": max_reward, "best_solutions": best_solutions}

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """We store the best solution and reward found."""
        max_rewards, best_solutions = outputs["max_reward"], outputs["best_solutions"]
        self.instance_rewards[batch_idx] = max_rewards
        self.instance_solutions[batch_idx, :] = best_solutions.squeeze(
            0
        )  # only one instance
        log.info(f"Best reward: {max_rewards.mean():.2f}")

    def on_train_epoch_end(self) -> None:
        """Called when the training ends.
        If the epoch ends, it means we have finished searching over the
        instances, thus the trainer should stop.
        """
        save_path = self.hparams.save_path
        if save_path is not None:
            log.info(f"Saving solutions and rewards to {save_path}...")
            torch.save(
                {"solutions": self.instance_solutions, "rewards": self.instance_rewards},
                save_path,
            )

        # https://github.com/Lightning-AI/lightning/issues/1406
        self.trainer.should_stop = True
