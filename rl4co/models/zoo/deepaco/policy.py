from functools import partial
from typing import Optional, Type

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.deepaco.antsystem import AntSystem
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.decoding import modify_logits_for_top_k_filtering, modify_logits_for_top_p_filtering
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.utils import merge_with_defaults


class DeepACOPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`. Introduced by Ye et al. (2023): https://arxiv.org/abs/2309.14032.
    This policy uses a Non-Autoregressive Graph Neural Network to generate heatmaps,
    which are then used to run Ant Colony Optimization (ACO) to construct solutions.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        env_name: Name of the environment used to initialize embeddings
        temperature: Temperature for the softmax during decoding. Defaults to 1.0.
        aco_class: Class representing the ACO algorithm to be used. Defaults to :class:`AntSystem`.
        aco_kwargs: Additional arguments to be passed to the ACO algorithm.
        train_with_local_search: Whether to train with local search. Defaults to False.
        n_ants: Number of ants to be used in the ACO algorithm. Can be an integer or dictionary. Defaults to 20.
        n_iterations: Number of iterations to run the ACO algorithm. Can be an integer or dictionary. Defaults to `dict(train=1, val=20, test=100)`.
        encoder_kwargs: Additional arguments to be passed to the encoder.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        train_with_local_search: bool = True,
        n_ants: Optional[int | dict] = None,
        n_iterations: Optional[int | dict] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(env_name=env_name, **encoder_kwargs)

        super(DeepACOPolicy, self).__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

        self.top_p = top_p
        self.top_k = top_k

        self.aco_class = AntSystem if aco_class is None else aco_class
        self.aco_kwargs = aco_kwargs
        self.train_with_local_search = train_with_local_search
        if train_with_local_search:
            assert self.aco_kwargs.get("use_local_search", False)
        self.n_ants = merge_with_defaults(n_ants, train=30, val=48, test=48)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)
        self.top_p = top_p
        self.top_k = top_k

    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[str | RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = True,
        return_hidden: bool = True,
        actions=None,
        **decoding_kwargs,
    ):
        """
        Forward method. During validation and testing, the policy runs the ACO algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_ants = self.n_ants[phase]
        # Instantiate environment if needed
        if (phase != "train" or self.train_with_local_search) and (
            env is None or isinstance(env, str)
        ):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)
        else:
            assert isinstance(env, RL4COEnvBase), "env must be an instance of RL4COEnvBase"

        if phase == "train":
            select_start_nodes_fn = partial(
                self.aco_class.select_start_node_fn,
                start_node=self.aco_kwargs.get("start_node", None),
            )
            decoding_kwargs.update(
                {
                    "select_start_nodes_fn": select_start_nodes_fn,
                    # TODO: Are they useful for training too?
                    # "top_p": self.top_p,
                    # "top_k": self.top_k,
                }
            )
            #  we just use the constructive policy
            outdict = super().forward(
                td_initial,
                env,
                phase=phase,
                decode_type="multistart_sampling",
                calc_reward=True,
                num_starts=n_ants,
                actions=actions,
                return_actions=return_actions,
                return_hidden=return_hidden,
                **decoding_kwargs,
            )

            outdict["reward"] = unbatchify(outdict["reward"], n_ants)

            if self.train_with_local_search:
                heatmap_logits = outdict["hidden"]
                # TODO: Refactor this so that we don't need to use the aco object
                aco = self.aco_class(heatmap_logits, n_ants=n_ants, **self.aco_kwargs)
                _, ls_reward = aco.local_search(
                    batchify(td_initial, n_ants), env, outdict["actions"]  # type:ignore
                )
                outdict["ls_reward"] = unbatchify(ls_reward, n_ants)

            outdict["log_likelihood"] = unbatchify(outdict["log_likelihood"], n_ants)
            return outdict

        heatmap_logits, _ = self.encoder(td_initial)
        heatmap_logits /= self.temperature

        if self.top_k > 0:
            self.top_k = min(self.top_k, heatmap_logits.size(-1))  # safety check
            heatmap_logits = modify_logits_for_top_k_filtering(heatmap_logits, self.top_k)

        if self.top_p > 0:
            assert self.top_p <= 1.0, "top-p should be in (0, 1]."
            heatmap_logits = modify_logits_for_top_p_filtering(heatmap_logits, self.top_p)

        aco = self.aco_class(heatmap_logits, n_ants=n_ants, **self.aco_kwargs)
        td, actions, reward = aco.run(td_initial, env, self.n_iterations[phase])

        out = {"reward": reward}
        if return_actions:
            out["actions"] = actions

        return out
