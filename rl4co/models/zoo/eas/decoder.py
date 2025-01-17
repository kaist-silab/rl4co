import math

import torch

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.utils.decoding import decode_logprobs, process_logits
from rl4co.utils.ops import batchify, unbatchify


def forward_pointer_attn_eas_lay(self, query, key, value, logit_key, mask):
    """Add layer to the forward pass of logit attention, i.e.
    Single-head attention.
    """
    # Compute inner multi-head attention with no projections.
    heads = self._inner_mha(query, key, value, mask)

    # Add residual for EAS layer if is set
    if getattr(self, "eas_layer", None) is not None:
        heads = heads + self.eas_layer(heads)

    glimpse = self.project_out(heads)

    # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
    # bmm is slightly faster than einsum and matmul
    logits = (
        torch.bmm(glimpse, logit_key.squeeze(1).transpose(-2, -1))
        / math.sqrt(glimpse.size(-1))
    ).squeeze(1)

    return logits


def forward_eas(
    self,
    td: TensorDict,
    cached_embeds,
    best_solutions,
    iter_count: int = 0,
    env: str | RL4COEnvBase = None,
    decode_type: str = "multistart_sampling",
    num_starts: int = None,
    mask_logits: bool = True,
    temperature: float = 1.0,
    tanh_clipping: float = 0,
    **decode_kwargs,
):
    """Forward pass of the decoder
    Given the environment state and the pre-computed embeddings, compute the logits and sample actions

    Args:
        td: Input TensorDict containing the environment state
        embeddings: Precomputed embeddings for the nodes. Can be already precomputed cached in form of q, k, v and
        env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
            it is more efficient to pass an already instantiated environment each time for fine-grained control
        decode_type: Type of decoding to use. Can be one of:
            - "sampling": sample from the logits
            - "greedy": take the argmax of the logits
            - "multistart_sampling": sample as sampling, but with multi-start decoding
            - "multistart_greedy": sample as greedy, but with multi-start decoding
        num_starts: Number of multi-starts to use. If None, will be calculated from the action mask
        calc_reward: Whether to calculate the reward for the decoded sequence
    """
    # TODO: this could be refactored by decoding strategies

    # Collect logprobs
    logprobs = []
    actions = []

    decode_step = 0
    # Multi-start decoding: first action is chosen by ad-hoc node selection
    if num_starts > 1 or "multistart" in decode_type:
        action = env.select_start_nodes(td, num_starts + 1) % num_starts
        # Append incumbent solutions
        if iter_count > 0:
            action = unbatchify(action, num_starts + 1)
            action[:, -1] = best_solutions[:, decode_step]
            action = action.permute(1, 0).reshape(-1)

        # Expand td to batch_size * (num_starts + 1)
        td = batchify(td, num_starts + 1)

        td.set("action", action)
        td = env.step(td)["next"]
        logp = torch.zeros_like(
            td["action_mask"], device=td.device
        )  # first logprobs is 0, so p = logprobs.exp() = 1

        logprobs.append(logp)
        actions.append(action)

    # Main decoding: loop until all sequences are done
    while not td["done"].all():
        decode_step += 1
        logits, mask = self.forward(td, cached_embeds, num_starts + 1)

        logp = process_logits(
            logits,
            mask,
            temperature=self.temperature if self.temperature is not None else temperature,
            tanh_clipping=(
                self.tanh_clipping if self.tanh_clipping is not None else tanh_clipping
            ),
            mask_logits=self.mask_logits if self.mask_logits is not None else mask_logits,
        )

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        action = decode_logprobs(logp, mask, decode_type=decode_type)

        if iter_count > 0:  # append incumbent solutions
            init_shp = action.shape
            action = unbatchify(action, num_starts + 1)
            action[:, -1] = best_solutions[:, decode_step]
            action = action.permute(1, 0).reshape(init_shp)

        td.set("action", action)
        td = env.step(td)["next"]

        # Collect output of step
        logprobs.append(logp)
        actions.append(action)

    logprobs, actions = torch.stack(logprobs, 1), torch.stack(actions, 1)
    rewards = env.get_reward(td, actions)
    return logprobs, actions, td, rewards
