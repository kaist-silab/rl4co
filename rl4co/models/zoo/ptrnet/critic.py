import torch
import torch.nn as nn

from .decoder import SimpleAttention
from .encoder import Encoder


class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        n_process_block_iters,
        tanh_exploration,
        use_tanh,
    ):
        super(CriticNetworkLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embed_dim, hidden_dim)

        self.process_block = SimpleAttention(
            hidden_dim, use_tanh=use_tanh, C=tanh_exploration
        )
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embed_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = (
            self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        )
        encoder_cx = (
            self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        )

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out
