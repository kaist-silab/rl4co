from functools import lru_cache

import torch

from tensordict import TensorDict

from rl4co.models.common.constructive.base import ConstructiveDecoder
from rl4co.utils.ops import batchify


@lru_cache(10)
def _multistart_batched_index(batch_size: int, num_starts: int):
    """Create a batched index for multistart decoding"""
    arr = torch.arange(batch_size)
    if num_starts <= 1:
        return arr
    else:
        return batchify(arr, num_starts)


class NonAutoregressiveDecoder(ConstructiveDecoder):
    """The nonautoregressive decoder is a simple callable class that
    takes the tensor dictionary and the heatmaps logits and returns the logits for the current
    action logits and the action mask.
    """

    def forward(self, td: TensorDict, heatmaps_logits: torch.Tensor, num_starts: int):
        return self.heatmap_to_logits(td, heatmaps_logits, num_starts)

    @staticmethod
    def heatmap_to_logits(td: TensorDict, heatmaps_logits: torch.Tensor, num_starts: int):
        """Obtain heatmap logits for current action to the next ones"""
        current_action = td.get("action", None)
        if current_action is None:
            logits = heatmaps_logits.mean(-1)
        else:
            batch_size = heatmaps_logits.shape[0]
            _indexer = _multistart_batched_index(batch_size, num_starts)
            logits = heatmaps_logits[_indexer, current_action, :]
        return logits, td["action_mask"]
