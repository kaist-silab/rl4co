from rl4co.models.common.constructive.base import ConstructivePolicy

from .decoder import NonAutoregressiveDecoder
from .encoder import NonAutoregressiveEncoder


class NonAutoregressivePolicy(ConstructivePolicy):
    """Template class for an nonautoregressive policy, simple wrapper around
    :class:`rl4co.models.common.constructive.base.ConstructivePolicy`.
    """

    def __init__(
        self,
        encoder: NonAutoregressiveEncoder,
        decoder: NonAutoregressiveDecoder = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        # If decoder is not passed, we default to the non-autoregressive decoder that decodes the heatmap
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super(NonAutoregressivePolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kw,
        )
