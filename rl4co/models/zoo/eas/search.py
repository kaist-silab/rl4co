import time

from functools import partial
from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rl4co.data.transforms import StateAugmentation
from rl4co.models.common.transductive import TransductiveModel
from rl4co.models.zoo.eas.decoder import forward_eas, forward_pointer_attn_eas_lay
from rl4co.models.zoo.eas.nn import EASLayerNet
from rl4co.utils.decoding import get_log_likelihood
from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EAS(TransductiveModel):
    """Efficient Active Search for Neural Combination Optimization from Hottung et al. (2022).
    Fine-tunes a subset of parameters (such as node embeddings or newly added layers) thus avoiding
    expensive re-encoding of the problem.
    Reference: https://openreview.net/pdf?id=nO5caZwFwYu

    Args:
        env: RL4CO environment to be solved
        policy: policy network
        dataset: dataset to be used for training
        use_eas_embedding: whether to use EAS embedding (EASEmb)
        use_eas_layer: whether to use EAS layer (EASLay)
        eas_emb_cache_keys: keys to cache in the embedding
        eas_lambda: lambda parameter for IL loss
        batch_size: batch size for training
        max_iters: maximum number of iterations
        augment_size: number of augmentations per state
        augment_dihedral: whether to augment with dihedral rotations
        parallel_runs: number of parallel runs
        baseline: REINFORCE baseline type (multistart, symmetric, full)
        max_runtime: maximum runtime in seconds
        save_path: path to save solution checkpoints
        optimizer: optimizer to use for training
        optimizer_kwargs: keyword arguments for optimizer
        verbose: whether to print progress for each iteration
    """

    def __init__(
        self,
        env,
        policy,
        dataset: Dataset | str,
        use_eas_embedding: bool = True,
        use_eas_layer: bool = False,
        eas_emb_cache_keys: list[str] = ["logit_key"],
        eas_lambda: float = 0.013,
        batch_size: int = 2,
        max_iters: int = 200,
        augment_size: int = 8,
        augment_dihedral: bool = True,
        num_parallel_runs: int = 1,
        baseline: str = "multistart",
        max_runtime: int = 86_400,
        save_path: str = None,
        optimizer: str | torch.optim.Optimizer | partial = "Adam",
        optimizer_kwargs: dict = {"lr": 0.0041, "weight_decay": 1e-6},
        verbose: bool = True,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        assert (
            self.hparams.use_eas_embedding or self.hparams.use_eas_layer
        ), "At least one of `use_eas_embedding` or `use_eas_layer` must be True."

        super(EAS, self).__init__(
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

        assert self.hparams.baseline in [
            "multistart",
            "symmetric",
            "full",
        ], f"Baseline {self.hparams.baseline} not supported."

    def setup(self, stage="fit"):
        """Setup base class and instantiate:
        - augmentation
        - instance solutions and rewards
        - original policy state dict
        """
        log.info(
            f"Setting up Efficient Active Search (EAS) with: \n"
            f"- EAS Embedding: {self.hparams.use_eas_embedding} \n"
            f"- EAS Layer: {self.hparams.use_eas_layer} \n"
        )
        super(EAS, self).setup(stage)

        # Instantiate augmentation
        self.augmentation = StateAugmentation(
            num_augment=self.hparams.augment_size,
            augment_fn="dihedral8" if self.hparams.augment_dihedral else "symmetric",
        )

        # Store original policy state dict
        self.original_policy_state = self.policy.state_dict()

        # Get dataset size and problem size
        len(self.dataset)
        _batch = next(iter(self.train_dataloader()))
        self.problem_size = self.env.reset(_batch)["action_mask"].shape[-1]
        self.instance_solutions = []
        self.instance_rewards = []

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Called before training (i.e. search) for a new batch begins.
        We re-load the original policy state dict and configure all parameters not to require gradients.
        We do the rest in the training step.
        """
        self.policy.load_state_dict(self.original_policy_state)

        # Set all policy parameters to not require gradients
        for param in self.policy.parameters():
            param.requires_grad = False

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
        num_instances = batch_size * n_aug * n_runs  # NOTE: no num_starts!
        # batch_r = n_runs * batch_size # effective batch size
        group_s = (
            n_start + 1
        )  # number of different rollouts per instance (+1 for incumbent solution construction)

        # Get encoder and decoder for simplicity
        encoder = self.policy.encoder
        decoder = self.policy.decoder

        # Precompute the cache of the embeddings (i.e. q,k,v and logit_key)
        embeddings, _ = encoder(td_init)
        cached_embeds = decoder._precompute_cache(embeddings)

        # Collect optimizer parameters
        opt_params = []
        if self.hparams.use_eas_layer:
            # EASLay: replace forward of logit attention computation. EASLayer
            eas_layer = EASLayerNet(num_instances, decoder.embed_dim).to(batch.device)
            decoder.pointer.eas_layer = partial(eas_layer, decoder.pointer)
            decoder.pointer.forward = partial(
                forward_pointer_attn_eas_lay, decoder.pointer
            )
            for param in eas_layer.parameters():
                opt_params.append(param)
        if self.hparams.use_eas_embedding:
            # EASEmb: set gradient of emb_key to True
            # for all the keys, wrap the embedding in a nn.Parameter
            for key in self.hparams.eas_emb_cache_keys:
                setattr(
                    cached_embeds, key, torch.nn.Parameter(getattr(cached_embeds, key))
                )
                opt_params.append(getattr(cached_embeds, key))
        decoder.forward_eas = partial(forward_eas, decoder)

        # We pass attributes saved in policy too
        def set_attr_if_exists(attr):
            if hasattr(self.policy, attr):
                setattr(decoder, attr, getattr(self.policy, attr))

        for attr in ["temperature", "tanh_clipping", "mask_logits"]:
            set_attr_if_exists(attr)

        self.configure_optimizers(opt_params)

        # Solution and reward buffer
        max_reward = torch.full((batch_size,), -float("inf"), device=batch.device)
        best_solutions = torch.zeros(
            batch_size, self.problem_size * 2, device=batch.device, dtype=int
        )  # i.e. incumbent solutions

        # Init search
        t_start = time.time()
        for iter_count in range(self.hparams.max_iters):
            # Evaluate policy with sampling multistarts passing the cached embeddings
            best_solutions_expanded = best_solutions.repeat(n_aug, 1).repeat(n_runs, 1)
            logprobs, actions, td_out, reward = decoder.forward_eas(
                td_init.clone(),
                cached_embeds=cached_embeds,
                best_solutions=best_solutions_expanded,
                iter_count=iter_count,
                env=self.env,
                decode_type="multistart_sampling",
                num_starts=n_start,
            )

            # Unbatchify to get correct dimensions
            ll = get_log_likelihood(logprobs, actions, td_out.get("mask", None))
            ll = unbatchify(ll, (n_runs * batch_size, n_aug, group_s)).squeeze()
            reward = unbatchify(reward, (n_runs * batch_size, n_aug, group_s)).squeeze()
            actions = unbatchify(actions, (n_runs * batch_size, n_aug, group_s)).squeeze()

            # Compute REINFORCE loss with shared baselines
            # compared to original EAS, we also support symmetric and full baselines
            group_reward = reward[..., :-1]  # exclude incumbent solution
            if self.hparams.baseline == "multistart":
                bl_val = group_reward.mean(dim=-1, keepdim=True)
            elif self.hparams.baseline == "symmetric":
                bl_val = group_reward.mean(dim=-2, keepdim=True)
            elif self.hparams.baseline == "full":
                bl_val = group_reward.mean(dim=-1, keepdim=True).mean(
                    dim=-2, keepdim=True
                )
            else:
                raise ValueError(f"Baseline {self.hparams.baseline} not supported.")

            # REINFORCE loss
            advantage = group_reward - bl_val
            loss_rl = -(advantage * ll[..., :-1]).mean()
            # IL loss
            loss_il = -ll[..., -1].mean()
            # Total loss
            loss = loss_rl + self.hparams.eas_lambda * loss_il

            # Manual backpropagation
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)

            # Save best solutions and rewards
            # Get max reward for each group and instance
            max_reward = reward.max(dim=2)[0].max(dim=1)[0]

            # Reshape and rank rewards
            reward_group = reward.reshape(n_runs * batch_size, -1)
            _, top_indices = torch.topk(reward_group, k=1, dim=1)

            # Obtain best solutions found so far
            solutions = actions.reshape(n_runs * batch_size, n_aug * group_s, -1)
            best_solutions_iter = gather_by_index(solutions, top_indices, dim=1)
            best_solutions[:, : best_solutions_iter.shape[1]] = best_solutions_iter

            self.log_dict(
                {
                    "loss": loss,
                    "max_reward": max_reward.mean(),
                    "step": iter_count,
                    "time": time.time() - t_start,
                },
                on_step=self.log_on_step,
            )

            log.info(
                f"{iter_count}/{self.hparams.max_iters} | "
                f" Reward: {max_reward.mean().item():.2f} "
            )

            # Stop if max runtime is exceeded
            if time.time() - t_start > self.hparams.max_runtime:
                log.info(f"Max runtime of {self.hparams.max_runtime} seconds exceeded.")
                break

        return {"max_reward": max_reward, "best_solutions": best_solutions}

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """We store the best solution and reward found."""
        max_rewards, best_solutions = outputs["max_reward"], outputs["best_solutions"]
        self.instance_solutions.append(best_solutions)
        self.instance_rewards.append(max_rewards)
        log.info(f"Best reward: {max_rewards.mean():.2f}")

    def on_train_epoch_end(self) -> None:
        """Called when the train ends."""
        save_path = self.hparams.save_path
        # concatenate solutions and rewards
        self.instance_solutions = pad_sequence(
            self.instance_solutions, batch_first=True, padding_value=0
        ).squeeze()
        self.instance_rewards = torch.cat(self.instance_rewards, dim=0).squeeze()
        if save_path is not None:
            log.info(f"Saving solutions and rewards to {save_path}...")
            torch.save(
                {"solutions": self.instance_solutions, "rewards": self.instance_rewards},
                save_path,
            )

        # https://github.com/Lightning-AI/lightning/issues/1406
        self.trainer.should_stop = True


class EASEmb(EAS):
    """EAS with embedding adaptation"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if not kwargs.get("use_eas_embedding", False) or kwargs.get(
            "use_eas_layer", True
        ):
            log.warning(
                "Setting `use_eas_embedding` to True and `use_eas_layer` to False. Use EAS base class to override."
            )
        kwargs["use_eas_embedding"] = True
        kwargs["use_eas_layer"] = False
        super(EASEmb, self).__init__(*args, **kwargs)


class EASLay(EAS):
    """EAS with layer adaptation"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if kwargs.get("use_eas_embedding", False) or not kwargs.get(
            "use_eas_layer", True
        ):
            log.warning(
                "Setting `use_eas_embedding` to True and `use_eas_layer` to False. Use EAS base class to override."
            )
        kwargs["use_eas_embedding"] = False
        kwargs["use_eas_layer"] = True
        super(EASLay, self).__init__(*args, **kwargs)
