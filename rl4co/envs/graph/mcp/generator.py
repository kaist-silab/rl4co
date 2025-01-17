from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def remove_repeat(x: torch.Tensor) -> torch.Tensor:
    """
    Remove the repeated elements in each row (i.e., the last dimension) of the input tensor x,
    and change the repeated elements to 0

    Ref: https://stackoverflow.com/questions/62300404

    Args:
        x: input tensor
    """

    # sorting the rows so that duplicate values appear together
    # e.g., first row: [1, 2, 3, 3, 3, 4, 4]
    y, indices = x.sort(dim=-1)

    # subtracting, so duplicate values will become 0
    # e.g., first row: [1, 2, 3, 0, 0, 4, 0]
    y[..., 1:] *= ((y[..., 1:] - y[..., :-1]) != 0).long()

    # retrieving the original indices of elements
    indices = indices.sort(dim=-1)[1]

    # re-organizing the rows following original order
    # e.g., first row: [1, 2, 3, 4, 0, 0, 0]
    return torch.gather(y, -1, indices)


class MCPGenerator(Generator):
    """Data generator for the Maximum Coverage Problem (MCP).

    Args:
        num_items: number of items in the MCP
        num_sets: number of sets in the MCP
        min_weight: minimum value for the item weights
        max_weight: maximum value for the item weights
        min_size: minimum size for the sets
        max_size: maximum size for the sets
        n_sets_to_choose: number of sets to choose in the MCP

    Returns:
        A TensorDict with the following keys:
            membership [batch_size, num_sets, max_size]: membership of items in sets
            weights [batch_size, num_items]: weights of the items
            n_sets_to_choose [batch_size, 1]: number of sets to choose in the MCP
    """

    def __init__(
        self,
        num_items: int = 200,
        num_sets: int = 100,
        min_weight: int = 1,
        max_weight: int = 10,
        min_size: int = 5,
        max_size: int = 15,
        n_sets_to_choose: int = 10,
        size_distribution: int | float | str | type | Callable = Uniform,
        weight_distribution: int | float | str | type | Callable = Uniform,
        **kwargs,
    ):
        self.num_items = num_items
        self.num_sets = num_sets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_size = min_size
        self.max_size = max_size
        self.n_sets_to_choose = n_sets_to_choose

        # Set size distribution
        if kwargs.get("size_sampler", None) is not None:
            self.size_sampler = kwargs["size_sampler"]
        else:
            self.size_sampler = get_sampler(
                "size", size_distribution, min_size, max_size + 1, **kwargs
            )

        # Item weight distribution
        if kwargs.get("weight_sampler", None) is not None:
            self.weight_sampler = kwargs["weight_sampler"]
        else:
            self.weight_sampler = get_sampler(
                "weight", weight_distribution, min_weight, max_weight + 1, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        try:
            batch_size = batch_size[0]
        except TypeError:
            batch_size = batch_size

        # Sample item weights
        weights_tensor = self.weight_sampler.sample((batch_size, self.num_items))
        weights_tensor = torch.floor(weights_tensor)
        weights_tensor = torch.clamp(weights_tensor, self.min_weight, self.max_weight)

        # Sample set sizes
        set_sizes = self.size_sampler.sample((batch_size, self.num_sets))
        set_sizes = torch.floor(set_sizes).long()
        set_sizes = torch.clamp(set_sizes, self.min_size, self.max_size)
        max_size = set_sizes.max().item()

        # Create membership tensor
        membership_tensor_max_size = torch.randint(
            1, self.num_items + 1, (batch_size, self.num_sets, max_size)
        )

        cutoffs_masks = torch.arange(self.max_size).view(1, 1, -1) < set_sizes.unsqueeze(
            -1
        )
        # Take the masked elements, 0 means the item is invalid
        membership_tensor = (
            membership_tensor_max_size * cutoffs_masks
        )  # (batch_size, num_sets, max_size)

        # Remove repeated items in each set
        membership_tensor = remove_repeat(membership_tensor)

        return TensorDict(
            {
                "membership": membership_tensor.float(),  # (batch_size, num_sets, max_size)
                "weights": weights_tensor.float(),  # (batch_size, num_items)
                "n_sets_to_choose": torch.ones(batch_size, 1)
                * self.n_sets_to_choose,  # (batch_size, 1)
            },
            batch_size=batch_size,
        )
