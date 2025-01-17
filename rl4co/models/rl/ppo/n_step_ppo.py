from typing import Any

import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Memory:
    def __init__(self):
        self.tds = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.tds[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


class n_step_PPO(RL4COLitModule):
    """
    An implementation of the n-step dactProximal Policy Optimization (PPO) algorithm (https://arxiv.org/abs/2110.02544)
    is presented for training improvement models.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        clip_range: float = 0.1,  # epsilon of PPO
        ppo_epochs: int = 3,  # inner epoch, K
        vf_lambda: float = 1.0,  # lambda of Value function fitting
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.05,  # max gradient norm
        gamma: float = 0.999,  # gamma for improvement MDP task
        n_step: float = 5,  # n-step for n-step PPO
        T_train: int = 250,  # the maximum inference T used for training
        T_test: int = 1000,  # the maximum inference T used for test
        lr_policy: float = 8e-5,  # the learning rate for actor
        lr_critic: float = 2e-5,  # the learning rate for critic
        CL_scalar: float = 2.0,  # hyperparameter of CL scalar of PPO-CL algorithm
        CL_best: bool = False,  # whether use the best solution from the CL rollout
        metrics: dict = {
            "train": ["loss", "surrogate_loss", "value_loss", "cost_bsf", "cost_init"],
            "val": ["cost_bsf", "cost_init"],
            "test": ["cost_bsf", "cost_init"],
        },
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs: dict = {
            "gamma": 0.985,  # the learning decay per epoch,
        },
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor=None,
        **kwargs,
    ):
        super().__init__(
            env,
            policy,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            **kwargs,
        )

        self.CL_scalar = CL_scalar
        self.CL_num = 0.0
        self.CL_best = CL_best
        self.automatic_optimization = False  # n_step_PPO uses custom optimization routine

        self.critic = critic

        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "vf_lambda": vf_lambda,
            "normalize_adv": normalize_adv,
            "max_grad_norm": max_grad_norm,
            "gamma": gamma,
            "n_step": n_step,
            "T_train": T_train,
            "T_test": T_test,
            "lr_policy": lr_policy,
            "lr_critic": lr_critic,
        }

    def configure_optimizers(self):
        parameters = [
            {"params": self.policy.parameters(), "lr": self.ppo_cfg["lr_policy"]}
        ] + [{"params": self.critic.parameters(), "lr": self.ppo_cfg["lr_critic"]}]

        return super().configure_optimizers(parameters)

    def on_train_epoch_end(self):
        """
        Learning rate scheduler and CL scheduler
        """
        # Learning rate scheduler
        sch = self.lr_schedulers()
        sch.step()

        # CL scheduler
        self.CL_num += 1 / self.CL_scalar

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        if phase != "train":
            with torch.no_grad():
                td = self.env.reset(batch)
                cost_init = td["cost_current"]
                for i in range(self.ppo_cfg["T_test"]):
                    out = self.policy(td, self.env, phase=phase)
                    self.env.step(td)
                out["cost_bsf"] = td["cost_bsf"]

        else:
            # init the training
            memory = Memory()
            td = self.env.reset(batch)

            # perform CL strategy
            with torch.no_grad():
                for i in range(int(self.CL_num)):
                    out = self.policy(td, self.env, phase=phase)
                    self.env.step(td)
            if self.CL_best:
                td = self.env.step_to_solution(td, td["rec_best"])
            cost_init = td["cost_current"]

            # perform gradiant updates every n_step untill reaching T_max
            assert (
                self.ppo_cfg["T_train"] % self.ppo_cfg["n_step"] == 0
            ), "T_max should be divided by n_step with no remainder"
            t = 0
            while t < self.ppo_cfg["T_train"]:
                memory.clear_memory()
                bl = []
                ll = []
                # Rollout for n_step,  perform actor and critic and env step, store the information in memory
                for i in range(self.ppo_cfg["n_step"]):
                    memory.tds.append(td.clone())

                    out = self.policy(td, self.env, phase=phase, return_embeds=True)
                    value_pred = self.critic(
                        out["embeds"].detach(), td["cost_bsf"].unsqueeze(-1)
                    )

                    memory.actions.append(out["actions"].clone())
                    memory.logprobs.append(out["log_likelihood"].clone())
                    bl.append(value_pred)

                    self.env.step(td)
                    memory.rewards.append(td["reward"].clone().view(-1, 1))

                t += self.ppo_cfg["n_step"]

                # PPO inner epoch, K
                old_value = None
                for k in range(self.ppo_cfg["ppo_epochs"]):
                    if k == 0:
                        ll = memory.logprobs

                    else:
                        ll = []
                        bl = []
                        for i in range(self.ppo_cfg["n_step"]):
                            out = self.policy(
                                memory.tds[i].clone(),
                                actions=memory.actions[i],
                                env=self.env,
                                phase=phase,
                                return_actions=False,
                                return_embeds=True,
                            )
                            bl_value = self.critic(
                                out["embeds"].detach(),
                                memory.tds[i]["cost_bsf"].unsqueeze(-1),
                            )

                            ll.append(out["log_likelihood"])
                            bl.append(bl_value)

                    # prepare loglikelihood (ll) and baseline value (bl)
                    ll = torch.stack(ll).view(-1, 1)
                    bl = torch.stack(bl).view(-1, 1)
                    old_ll = torch.stack(memory.logprobs).view(-1, 1)

                    # Compute the Reward wrt n_step
                    Reward = []
                    reward_reversed = memory.rewards[::-1]
                    R = self.critic(
                        self.policy(td, self.env, phase=phase, only_return_embed=True)[
                            "embeds"
                        ].detach(),
                        td["cost_bsf"].unsqueeze(-1),
                    ).detach()  # Remember to detach() since we only need the predicted value here
                    for r in range(len(reward_reversed)):
                        R = R * self.ppo_cfg["gamma"] + reward_reversed[r]
                        Reward.append(R.clone())
                    Reward = torch.stack(Reward[::-1]).view(-1, 1)

                    # Compute the ratio of probabilities of new and old actions
                    ratio = torch.exp(ll - old_ll.detach())

                    # Compute the advantage
                    adv = Reward - bl.detach()

                    # Normalize advantage
                    if self.ppo_cfg["normalize_adv"]:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Compute the surrogate loss
                    surrogate_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(
                            ratio,
                            1 - self.ppo_cfg["clip_range"],
                            1 + self.ppo_cfg["clip_range"],
                        )
                        * adv,
                    ).mean()

                    # compute value function loss
                    if old_value is None:
                        value_loss = ((bl - Reward) ** 2).mean()
                        old_value = bl.detach()
                    else:
                        value_clipped = (
                            torch.clamp(
                                bl - old_value,
                                -self.ppo_cfg["clip_range"],
                                self.ppo_cfg["clip_range"],
                            )
                            + old_value
                        )

                        value_loss = torch.max(
                            (bl - Reward) ** 2,
                            (value_clipped - Reward) ** 2,
                        ).mean()

                    # compute total loss
                    loss = surrogate_loss + self.ppo_cfg["vf_lambda"] * value_loss

                    # perform manual optimization following the Lightning routine
                    # https://lightning.ai/docs/pytorch/stable/common/optimization.html
                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    if self.ppo_cfg["max_grad_norm"] is not None:
                        self.clip_gradients(
                            opt,
                            gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                            gradient_clip_algorithm="norm",
                        )
                    opt.step()

            out.update(
                {
                    "cost_init": cost_init,
                    "cost_bsf": td["cost_bsf"],
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": value_loss,
                }
            )
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
