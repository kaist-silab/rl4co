import pytest
import torch

from tensordict import TensorDict
from torch.nn.functional import scaled_dot_product_attention

from rl4co.models.nn.attention import scaled_dot_product_attention_simple
from rl4co.utils.decoding import process_logits
from rl4co.utils.ops import batchify, unbatchify


@pytest.mark.parametrize(
    "a",
    [
        torch.randn(10, 20, 2),
        TensorDict(
            {"a": torch.randn(10, 20, 2), "b": torch.randn(10, 20, 2)}, batch_size=10
        ),
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_batchify(a, shape):
    # batchify: [b, ...] -> [b * prod(shape), ...]
    # unbatchify: [b * prod(shape), ...] -> [b, shape[0], shape[1], ...]
    a_batch = batchify(a, shape)
    a_unbatch = unbatchify(a_batch, shape)
    if isinstance(a, TensorDict):
        a, a_unbatch = a["a"], a_unbatch["a"]
    index = (slice(None),) + (0,) * len(shape)  # (slice(None), 0, 0, ..., 0)
    assert torch.allclose(a, a_unbatch[index])


@pytest.mark.parametrize("top_p", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("top_k", [0, 5, 10])
def test_top_k_top_p_sampling(top_p, top_k):
    logits = torch.randn(8, 10)
    mask = torch.ones(8, 10).bool()
    logprobs = process_logits(logits, mask, top_p=top_p, top_k=top_k)
    assert len(logprobs) == logits.size(0)


def test_scaled_dot_product_attention():
    bs, ns, ds = 2, 3, 4
    q = torch.rand(bs, ns, ds)
    k = torch.rand(bs, ns, ds)
    v = torch.rand(bs, ns, ds)
    attn_mask = torch.rand(bs, ns, ns) > 0.5
    attn_mask[:, 0, :] = True  # at least one row element is True
    attn_mask[:, :, 0] = True  # at least one column element is True
    attn_torch = scaled_dot_product_attention(q, k, v, attn_mask)
    attn_rl4co = scaled_dot_product_attention_simple(q, k, v, attn_mask)
    assert torch.allclose(attn_torch, attn_rl4co)
