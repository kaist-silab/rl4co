from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.zoo.mdam.decoder import MDAMDecoder
from rl4co.models.zoo.mdam.encoder import MDAMGraphAttentionEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MDAMPolicy(AutoregressivePolicy):
    """Multi-Decoder Attention Model (MDAM) policy.
    Args:

    """

    def __init__(
        self,
        encoder: MDAMGraphAttentionEncoder = None,
        decoder: MDAMDecoder = None,
        embed_dim: int = 128,
        env_name: str = "tsp",
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        **decoder_kwargs,
    ):
        encoder = (
            MDAMGraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
            )
            if encoder is None
            else encoder
        )

        decoder = (
            MDAMDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_heads=num_heads,
                **decoder_kwargs,
            )
            if decoder is None
            else decoder
        )

        super(MDAMPolicy, self).__init__(
            env_name=env_name, encoder=encoder, decoder=decoder
        )

        self.init_embedding = env_init_embedding(env_name, {"embed_dim": embed_dim})

    def forward(
        self,
        td: TensorDict,
        env: str | RL4COEnvBase = None,
        phase: str = "train",
        return_actions: bool = True,
        **decoder_kwargs,
    ) -> TensorDict:
        embedding = self.init_embedding(td)
        encoded_inputs, _, attn, V, h_old = self.encoder(embedding)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        reward, log_likelihood, kl_divergence, actions = self.decoder(
            td, encoded_inputs, env, attn, V, h_old, self.encoder, **decoder_kwargs
        )
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "entropy": kl_divergence,
            "actions": actions if return_actions else None,
        }
        return out
