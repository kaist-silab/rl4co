import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.models.zoo.polynet.decoder import PolyNetDecoder


class PolyNetPolicy(AutoregressivePolicy):
    """
    # TODO
    Polynet policy based on Hottung et al. (2024) https://arxiv.org/abs/2402.14048.
    The model uses either the AttentionModel encoder or the MatNet encoder in combination with
    a custom PolyNet decoder.

    Note: The default arguments for the AttentionModel encoder follow the POMO paper. The default decoding type
    during validation and testing is 'sampling'.

    Args:
        k: Number of strategies to learn ("K" in the paper)
        encoder_type: Type of encoder that should be used. "AM" or "MatNet" are supported.
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        mask_logits: Whether to mask the logits during decoding
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
        **kwargs: keyword arguments passed to the encoder and decoder modules
    """

    def __init__(
        self,
        k: int,
        encoder: nn.Module = None,
        encoder_type: str = "AM",
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str | RL4COEnvBase = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
        **kwargs,
    ):
        if encoder is None:
            if encoder_type == "AM":
                encoder = AttentionModelEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_encoder_layers,
                    env_name=env_name,
                    normalization=normalization,
                    feedforward_hidden=feedforward_hidden,
                    **kwargs,
                )
            elif encoder_type == "MatNet":
                kwargs_with_defaults = {"init_embedding_kwargs": {"mode": "RandomOneHot"}}
                kwargs_with_defaults.update(kwargs)
                encoder = MatNetEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_encoder_layers,
                    normalization=normalization,
                    **kwargs_with_defaults,
                )

        decoder = PolyNetDecoder(
            k=k,
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            **kwargs,
        )

        super(PolyNetPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **kwargs,
        )
