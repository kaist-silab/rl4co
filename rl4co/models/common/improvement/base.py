import abc

from typing import Tuple

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.pos_embeddings import pos_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ImprovementEncoder(nn.Module):
    """Base class for the encoder of improvement models"""

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        env_name: str = "pdp_ruin_repair",
        pos_type: str = "CPE",
        num_heads: int = 4,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 128,
        linear_bias: bool = False,
    ):
        super(ImprovementEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.init_embedding = (
            env_init_embedding(
                self.env_name, {"embed_dim": embed_dim, "linear_bias": linear_bias}
            )
            if init_embedding is None
            else init_embedding
        )

        self.pos_type = pos_type
        self.pos_embedding = (
            pos_init_embedding(self.pos_type, {"embed_dim": embed_dim})
            if pos_embedding is None
            else pos_embedding
        )

    @abc.abstractmethod
    def _encoder_forward(self, init_h: Tensor, init_p: Tensor) -> Tuple[Tensor, Tensor]:
        """Process the node embeddings and positional embeddings to the final embeddings

        Args:
            init_h: initialized node embeddings
            init_p: initialized positional embeddings

        Returns:
            Tuple containing the final node embeddings and final positional embeddings (if any)
        """
        raise NotImplementedError("Implement me in subclass!")

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space (node)
        init_h = self.init_embedding(td)

        # Transfer to embedding space (solution)
        init_p = self.pos_embedding(td)

        # Process embedding
        final_h, final_p = self._encoder_forward(init_h, init_p)

        # Return latent representation and initial embedding
        return final_h, final_p


class ImprovementDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for improvement models. The decoder is responsible for generating the logits of the action"""

    @abc.abstractmethod
    def forward(self, td: TensorDict, final_h: Tensor, final_p: Tensor) -> Tensor:
        """Obtain logits to perform operators that improve the current solution to the next ones

        Args:
            td: TensorDict with the current environment state
            final_h: final node embeddings
            final_p: final positional embeddings

        Returns:
            Tuple containing the logits
        """
        raise NotImplementedError("Implement me in subclass!")


class ImprovementPolicy(nn.Module):
    """
    Base class for improvement policies. Improvement policies take an instance + a solution as input and output a specific operator that changes the current solution to a new one.

    "Improvement" means that a solution is (potentially) improved to a new one by the model.

    """

    @abc.abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: str | RL4COEnvBase = None,
        phase: str = "train",
        return_actions: bool = True,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_init_embeds: Whether to return the initial embeddings
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """
        raise NotImplementedError("Implement me in subclass!")
