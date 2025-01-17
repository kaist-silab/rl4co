import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.utils.decoding import decode_logprobs
from rl4co.utils.ops import gather_by_index


class SimpleAttention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(SimpleAttention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, F.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * F.tanh(u)
        else:
            logits = u
        return e, logits


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        tanh_exploration: float = 10.0,
        use_tanh: bool = True,
        num_glimpses=1,
        mask_glimpses=True,
        mask_logits=True,
    ):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_glimpses = num_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration

        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.pointer = SimpleAttention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = SimpleAttention(hidden_dim, use_tanh=False)

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), False)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):
        logit_mask = (
            self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        )

        logits, h_out = self.calc_logits(
            x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits
        )

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)

        if not self.mask_logits:
            log_p[~logit_mask] = float("-inf")

        return h_out, log_p, logit_mask

    def calc_logits(
        self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None
    ):
        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.num_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[~logit_mask] = float("-inf")
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[~logit_mask] = float("-inf")

        return logits, h_out

    def forward(
        self,
        decoder_input,
        embedded_inputs,
        hidden,
        context,
        decode_type="sampling",
        eval_tours=None,
    ):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embed_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embed_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = torch.ones(
            embedded_inputs.size(1),
            embedded_inputs.size(0),
            dtype=torch.bool,
            device=embedded_inputs.device,
        )

        for i in steps:
            hidden, log_p, mask = self.recurrence(
                decoder_input, hidden, mask, idxs, i, context
            )
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = (
                decode_logprobs(log_p, mask, decode_type=decode_type)
                if eval_tours is None
                else eval_tours[:, i]
            )
            # select logp of chosen action
            log_p = gather_by_index(log_p, idxs, dim=1)

            idxs = (
                idxs.detach()
            )  # Otherwise pytorch complains it want's a reward, todo implement this more properly?
            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous()
                .view(1, batch_size, 1)
                .expand(1, batch_size, *embedded_inputs.size()[2:]),
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)
        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden
