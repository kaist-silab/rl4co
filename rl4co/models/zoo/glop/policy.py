from typing import Optional, Union
from uu import decode

from einops import rearrange
import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressiveDecoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

log = get_pylogger(__name__)


class GLOPPolicy(NonAutoregressivePolicy):
    """Global and Local Optimization Policies (GLOP) Policy: https://arxiv.org/abs/2312.08224

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        revisers: List of revisers to use for the GLOP revision phase, the reviser could be a neural network model
            or a heuristic function. Defaults to None, but this is required.
        n_samples: Number of samples to use for the GLOP policy. Defaults to 10.
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`
    """

    def __init__(
        self,
        encoder: NonAutoregressiveEncoder = None,
        decoder: NonAutoregressiveDecoder = None,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        n_samples: int = 10,
        revisers: list = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(**encoder_kwargs)
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

        self.n_samples = n_samples
        self.revisers = revisers

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        return_partitions: bool = True,
        return_partitions_actions: bool = True,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        device = td.device
        
        par_out = super().forward(
            td = td,
            env = env,
            phase = phase,
            calc_reward = False, # We don't need the partition reward
            return_actions = True, # Used for partition
            return_entropy = return_entropy,
            return_init_embeds = return_init_embeds,
            return_sum_log_likelihood = return_sum_log_likelihood,
            num_starts = self.n_samples,
            actions = actions,
            decode_type="multistart_sampling",
            **decoding_kwargs,
        )

        td_sample = batchify(td, self.n_samples)
        par_actions = par_out["actions"]
        par_log_likelihood = par_out["log_likelihood"]

        # Based on partition actions to get partitions
        shpp_locs, par = self.partition(td_sample, par_actions)

        # Batchify the shpp_td along the partitions
        batch_size = shpp_locs.size(0)
        n_partitions = shpp_locs.size(1)
        n_nodes = shpp_locs.size(2)
        shpp_locs = rearrange(shpp_locs, "b p n d -> (b p) n d", b=batch_size, p=n_partitions, n=n_nodes, d=2)

        # Set the SHPP environments
        shpp_env = get_env("shpp")
        shpp_env.generator.num_loc = n_nodes
        shpp_td = shpp_env.reset(batch_size=batch_size*n_partitions).to(device)
        shpp_td.set("locs", shpp_locs)

        # Call revisers to solve the sub-routes and record the best
        best_revised_reward = torch.full(shpp_td.shape[:1], float("-inf")).to(device)
        best_revised_actions = torch.zeros(shpp_td["locs"].shape[:-1], dtype=torch.int64).to(device)
        for reviser in self.revisers:
            reviser = reviser.to(device)
            reviser_out = reviser(shpp_td.clone(), phase="test", decode_type="greedy", return_actions=True)

            # Record the best
            improve_flag = reviser_out["reward"] > best_revised_reward
            best_revised_reward = torch.where(improve_flag, reviser_out["reward"], best_revised_reward)
            best_revised_actions = torch.where(improve_flag.unsqueeze(1), reviser_out["actions"], best_revised_actions)

        # Construct final output
        out = {"log_likelihood": par_log_likelihood}

        if calc_reward:
            best_revised_reward = unbatchify(best_revised_reward, (n_partitions))
            best_revised_reward = best_revised_reward.sum(dim=-1)
            out["reward"] = best_revised_reward
        if return_actions:
            final_actions = unbatchify(best_revised_actions, (n_partitions))
            final_actions = final_actions.flatten(start_dim=1)
            out["actions"] = final_actions
        if return_entropy:
            out["entropy"] = par_out["entropy"]
        if return_init_embeds:
            out["init_embeds"] = par_out["init_embeds"]
        if return_partitions:
            out["partition"] = par
        if return_partitions_actions:
            out["par_actions"] = par_actions
            out["revised_actions"] = best_revised_actions

        return out

    @staticmethod
    def partition(td: TensorDict, actions: torch.Tensor):
        """
        Args:
            td [bs*n_samples]
            actions [bs*n_samples, seq_len]
        Returns:
            <TensorDict>
                locs [bs*n_samples, n_partitions, n_nodes, 2]
                partition [bs*n_samples, n_partitions, seq_len]
        """
        max_num_partitions = 0
        max_len_sequence = 0
        partition = torch.zeros([*actions.size(), actions.size(-1)]).to(
            td.device, torch.int64
        )  # [bs*n_samples, seq_len, seq_len]
        for batch_idx in range(td.size(0)):
            partition_idx = 0
            partition_start_idx = 0
            for action_idx, action in enumerate(actions[batch_idx]):
                if (action == 0) & (action_idx != 0):
                    partition_idx += 1
                    # Update the max length of the sequence
                    if action_idx - partition_start_idx > max_len_sequence:
                        max_len_sequence = action_idx - partition_start_idx
                    partition_start_idx = action_idx + 1
                else:
                    partition[
                        batch_idx, partition_idx, action_idx - partition_start_idx
                    ] = action
            # Update the max number of partitions
            if partition_idx + 1 > max_num_partitions:
                max_num_partitions = partition_idx + 1
        # Squeese the partition
        partition = partition[:, :max_num_partitions, :max_len_sequence]
        # Adding depot to the beginning and the end
        partition = torch.cat(
            [
                torch.zeros_like(partition[:, :, :1]),
                partition,
                torch.zeros_like(partition[:, :, :1]),
            ],
            dim=-1,
        )
        # Expand the locs
        locs = td["locs"].unsqueeze(1).expand(-1, max_num_partitions, -1, -1)
        # Get the locations of the partitions
        locs = gather_by_index(locs, partition, dim=-2)

        return locs, partition
