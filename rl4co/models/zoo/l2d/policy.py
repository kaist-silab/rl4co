from typing import Optional

import torch
import torch.nn as nn

from torch.distributions import Categorical

from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveDecoder,
    AutoregressiveEncoder,
    AutoregressivePolicy,
)
from rl4co.models.common.constructive.base import NoEncoder
from rl4co.models.nn.env_embeddings.init import FJSPMatNetInitEmbedding
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.models.nn.mlp import MLP
from rl4co.models.zoo.matnet.matnet_w_sa import Encoder
from rl4co.utils.decoding import DecodingStrategy, process_logits
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .decoder import L2DAttnActor, L2DDecoder
from .encoder import GCN4JSSP

log = get_pylogger(__name__)


class L2DPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjsp",
        het_emb: bool = True,
        scaling_factor: int = 1000,
        normalization: str = "batch",
        init_embedding: Optional[nn.Module] = None,
        stepwise_encoding: bool = False,
        tanh_clipping: float = 10,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            if stepwise_encoding:
                encoder = NoEncoder()
            elif env_name == "fjsp" or (env_name == "jssp" and het_emb):
                encoder = HetGNNEncoder(
                    env_name=env_name,
                    embed_dim=embed_dim,
                    num_layers=num_encoder_layers,
                    normalization="batch",
                    init_embedding=init_embedding,
                    scaling_factor=scaling_factor,
                )
            else:
                encoder = GCN4JSSP(
                    embed_dim,
                    num_encoder_layers,
                    init_embedding=init_embedding,
                    scaling_factor=scaling_factor,
                )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = L2DDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                actor_hidden_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                init_embedding=init_embedding,
                het_emb=het_emb,
                stepwise=stepwise_encoding,
                scaling_factor=scaling_factor,
                normalization=normalization,
            )

        # Pass to constructive policy
        super(L2DPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            tanh_clipping=tanh_clipping,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )


class L2DAttnPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        scaling_factor: int = 1000,
        normalization: str = "batch",
        env_name: str = "fjsp",
        init_embedding: Optional[nn.Module] = None,
        tanh_clipping: float = 10,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            if init_embedding is None:
                init_embedding = FJSPMatNetInitEmbedding(
                    embed_dim, scaling_factor=scaling_factor
                )

            encoder = Encoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=embed_dim * 2,
                init_embedding=init_embedding,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = L2DAttnActor(
                env_name=env_name,
                embed_dim=embed_dim,
                num_heads=num_heads,
                scaling_factor=scaling_factor,
                stepwise=False,
            )

        # Pass to constructive policy
        super(L2DAttnPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            tanh_clipping=tanh_clipping,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )


class L2DPolicy4PPO(L2DPolicy):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        critic=None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjsp",
        het_emb: bool = True,
        scaling_factor: int = 1000,
        init_embedding=None,
        tanh_clipping: float = 10,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if init_embedding is None:
            pass  # TODO PPO specific init emb?

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            env_name=env_name,
            het_emb=het_emb,
            scaling_factor=scaling_factor,
            init_embedding=init_embedding,
            stepwise_encoding=True,
            tanh_clipping=tanh_clipping,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )

        if critic is None:
            if env_name == "fjsp" or het_emb:
                input_dim = 2 * embed_dim
            else:
                input_dim = embed_dim
            critic = MLP(input_dim, 1, num_neurons=[embed_dim] * 2)

        self.critic = critic
        assert isinstance(
            self.encoder, NoEncoder
        ), "Define a feature extractor for decoder rather than an encoder in stepwise PPO"

    def evaluate(self, td):
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, _ = self.decoder.feature_extractor(td)
        # pool the embeddings for the critic
        h_tuple = (hidden,) if isinstance(hidden, torch.Tensor) else hidden
        pooled = tuple(map(lambda x: x.mean(dim=-2), h_tuple))
        # potentially cat multiple embeddings (pooled ops and machines)
        h_pooled = torch.cat(pooled, dim=-1)
        # pred value via the value head
        value_pred = self.critic(h_pooled)
        # pre decoder / actor hook
        td, _, hidden = self.decoder.actor.pre_decoder_hook(
            td, None, hidden, num_starts=0
        )
        logits, mask = self.decoder.actor(td, *hidden)
        # get logprobs and entropy over logp distribution
        logprobs = process_logits(logits, mask, tanh_clipping=self.tanh_clipping)
        action_logprobs = gather_by_index(logprobs, td["action"], dim=1)
        dist_entropys = Categorical(logprobs.exp()).entropy()

        return action_logprobs, value_pred, dist_entropys

    def act(self, td, env, phase: str = "train"):
        logits, mask = self.decoder(td, hidden=None, num_starts=0)
        logprobs = process_logits(logits, mask, tanh_clipping=self.tanh_clipping)

        # DRL-S, sampling actions following \pi
        if phase == "train":
            action_indexes = DecodingStrategy.sampling(logprobs)
            td["logprobs"] = gather_by_index(logprobs, action_indexes, dim=1)

        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = DecodingStrategy.greedy(logprobs)

        # memories.states.append(copy.deepcopy(state))
        td["action"] = action_indexes

        return td

    @torch.no_grad()
    def generate(self, td, env=None, phase: str = "train", **kwargs) -> dict:
        assert phase != "train", "dont use generate() in training mode"
        with torch.no_grad():
            out = super().__call__(td, env, phase=phase, **kwargs)
        return out
