import math

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.improvement.base import ImprovementPolicy
from rl4co.models.zoo.n2s.encoder import N2SEncoder
from rl4co.models.zoo.neuopt.decoder import RDSDecoder
from rl4co.utils.decoding import DecodingStrategy, get_decoding_strategy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CustomizeTSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(CustomizeTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Sequential(
            nn.Linear(node_dim, embed_dim // 2, linear_bias),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim, linear_bias),
        )

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class NeuOptPolicy(ImprovementPolicy):
    """
    NeuOpt Policy based on Ma et al. (2023)
    This model first encodes the input graph and current solution using a N2S encoder (:class:`N2SEncoder`)
    and then decodes the k-opt action (:class:`RDSDecoder`)

    Args:
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        pos_type: Name of the used positional encoding method (CPE or APE)
        init_embedding: Module to use for the initialization of the embeddings
        pos_embedding: Module to use for the initialization of the positional embeddings
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 4,
        normalization: str = "layer",
        feedforward_hidden: int = 128,
        env_name: str = "tsp_kopt",
        pos_type: str = "CPE",
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        temperature: float = 1.0,
        tanh_clipping: float = 6.0,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
    ):
        super(NeuOptPolicy, self).__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # Encoder and decoder
        if init_embedding is None:
            init_embedding = CustomizeTSPInitEmbedding(self.embed_dim)

        self.encoder = N2SEncoder(
            embed_dim=embed_dim,
            init_embedding=init_embedding,
            pos_embedding=pos_embedding,
            env_name=env_name,
            pos_type=pos_type,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        self.decoder = RDSDecoder(embed_dim=embed_dim)

        self.init_hidden_W = nn.Linear(self.embed_dim, self.embed_dim)
        self.init_query_learnable = nn.Parameter(torch.Tensor(self.embed_dim))

        self.init_parameters()

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        td: TensorDict,
        env: str | RL4COEnvBase = None,
        phase: str = "train",
        return_actions: bool = True,
        return_embeds: bool = False,
        only_return_embed: bool = False,
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
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        nfe, _ = self.encoder(td)
        if only_return_embed:
            return {"embeds": nfe.detach()}

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)
        assert not env.two_opt_mode, "NeuOpt only support k-opt with k > 2"

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=True,
            improvement_method_mode=True,
            **decoding_kwargs,
        )

        # Perform the decoding
        bs, gs, _, ll, action_sampled, rec, visited_time = (
            *nfe.size(),
            0.0,
            None,
            td["rec_current"],
            td["visited_time"],
        )
        action_index = torch.zeros(bs, env.k_max, dtype=torch.long).to(rec.device)
        k_action_left = torch.zeros(bs, env.k_max + 1, dtype=torch.long).to(rec.device)
        k_action_right = torch.zeros(bs, env.k_max, dtype=torch.long).to(rec.device)
        next_of_last_action = (
            torch.zeros_like(rec[:, :1], dtype=torch.long).to(rec.device) - 1
        )
        mask = torch.zeros_like(rec, dtype=torch.bool).to(rec.device)
        stopped = torch.ones(bs, dtype=torch.bool).to(rec.device)
        zeros = torch.zeros((bs, 1), device=td.device)

        # init queries
        h_mean = nfe.mean(1)
        init_query = self.init_query_learnable.repeat(bs, 1)
        input_q1 = input_q2 = init_query.clone()
        init_hidden = self.init_hidden_W(h_mean)
        q1 = q2 = init_hidden.clone()

        for i in range(env.k_max):
            # Pass RDS decoder
            logits, q1, q2 = self.decoder(nfe, q1, q2, input_q1, input_q2)

            # Calc probs
            if i == 0 and "action" in td.keys():
                mask = mask.scatter(1, td["action"][:, :1], 1)

            logprob, action_sampled = decode_strategy.step(
                logits,
                ~mask.clone(),
                action=actions[:, i : i + 1].squeeze() if actions is not None else None,
            )
            action_sampled = action_sampled.unsqueeze(-1)
            if i > 0:
                action_sampled = torch.where(
                    stopped.unsqueeze(-1), action_index[:, :1], action_sampled
                )
            if phase == "train":
                loss_now = logprob.gather(1, action_sampled)
            else:
                loss_now = zeros.clone()

            # Record log_likelihood and Entropy
            if i > 0:
                ll = ll + torch.where(stopped.unsqueeze(-1), zeros * 0, loss_now)
            else:
                ll = ll + loss_now

            # Store and Process actions
            next_of_new_action = rec.gather(1, action_sampled)
            action_index[:, i] = action_sampled.squeeze().clone()
            k_action_left[stopped, i] = action_sampled[stopped].squeeze().clone()
            k_action_right[~stopped, i - 1] = action_sampled[~stopped].squeeze().clone()
            k_action_left[:, i + 1] = next_of_new_action.squeeze().clone()

            # Prepare next RNN input
            input_q1 = nfe.gather(
                1, action_sampled.view(bs, 1, 1).expand(bs, 1, self.embed_dim)
            ).squeeze(1)
            input_q2 = torch.where(
                stopped.view(bs, 1).expand(bs, self.embed_dim),
                input_q1.clone(),
                nfe.gather(
                    1,
                    (next_of_last_action % gs)
                    .view(bs, 1, 1)
                    .expand(bs, 1, self.embed_dim),
                ).squeeze(1),
            )

            # Process if k-opt close
            # assert (input_q1[stopped] == input_q2[stopped]).all()
            if i > 0:
                stopped = stopped | (action_sampled == next_of_last_action).squeeze()
            else:
                stopped = (action_sampled == next_of_last_action).squeeze()
            # assert (input_q1[stopped] == input_q2[stopped]).all()

            k_action_left[stopped, i] = k_action_left[stopped, i - 1]
            k_action_right[stopped, i] = k_action_right[stopped, i - 1]

            # Calc next basic masks
            if i == 0:
                visited_time_tag = (
                    visited_time - visited_time.gather(1, action_sampled)
                ) % gs
            mask &= False
            mask[(visited_time_tag <= visited_time_tag.gather(1, action_sampled))] = True
            if i == 0:
                mask[visited_time_tag > (gs - 2)] = True
            mask[stopped, action_sampled[stopped].squeeze()] = (
                False  # allow next k-opt starts immediately
            )
            # if True:#i == env.k_max - 2: # allow special case: close k-opt at the first selected node
            index_allow_first_node = (~stopped) & (
                next_of_new_action.squeeze() == action_index[:, 0]
            )
            mask[index_allow_first_node, action_index[index_allow_first_node, 0]] = False

            # Move to next
            next_of_last_action = next_of_new_action
            next_of_last_action[stopped] = -1

        # Form final action
        k_action_right[~stopped, -1] = k_action_left[~stopped, -1].clone()
        k_action_left = k_action_left[:, : env.k_max]
        action_all = torch.cat((action_index, k_action_left, k_action_right), -1)

        outdict = {"log_likelihood": ll, "cost_bsf": td["cost_bsf"]}
        td.set("action", action_all)

        if return_embeds:
            outdict["embeds"] = nfe.detach()

        if return_actions:
            outdict["actions"] = action_all

        return outdict
