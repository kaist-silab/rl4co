import torch

try:
    # from fla.ops.linear_attn.chunk_fuse import fused_chunk_linear_attn
    from fla.ops.linear_attn.chunk import chunk_linear_attn as fused_chunk_linear_attn
except ImportError:
    fused_chunk_linear_attn = None

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def fused_chunk_linear_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = True,
    **kwargs,
):
    assert (
        fused_chunk_linear_attn is not None
    ), "fused_chunk_linear_attn not found. Install Flash Linear Attention using instructions from https://github.com/sustcsonglin/flash-linear-attention"
    assert (
        kwargs.get("attn_mask", None) is None
    ), "attn_mask is not supported in Flash  Linear Attention"
    return fused_chunk_linear_attn(
        q, k, v, scale, initial_state, output_final_state, normalize
    )[0]


def scaled_dot_product_attention_flash_attn(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """
    Flash Attention 2 wrapper (https://github.com/Dao-AILab/flash-attention) around `flash_attn_func` to obtain the same behavior as
    `torch.nn.functional.scaled_dot_product_attention`.
    We need to permute the query, key, and value tensors before calling the scaled dot product attention function
    Reference: https://github.com/Dao-AILab/flash-attention/issues/383

    Note:
        Flash Attention does not support masking except for causal masking.

    Args:
        q (torch.Tensor): Query tensor of shape `(batch_size, num_heads, seq_len_q, head_dim)`
        k (torch.Tensor): Key tensor of shape `(batch_size, num_heads, seq_len_k, head_dim)`
        v (torch.Tensor): Value tensor of shape `(batch_size, num_heads, seq_len_v, head_dim)`
        attn_mask (torch.Tensor): Attention mask of shape `(batch_size, seq_len_q, seq_len_k)`
        dropout_p (float): Dropout probability
        is_causal (bool): Whether to apply causal mask to attention scores
    """
    assert attn_mask is None, "`attn_mask` is not supported in Flash Attention"
    assert flash_attn_func is not None, (
        "Flash Attention not found. Install Flash Attention using instructions from "
        "https://github.com/Dao-AILab/flash-attention . "
        "Alternatively, use `torch.nn.functional.scaled_dot_product_attention` available from PyTorch 2.0.0"
    )
    q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    return out.transpose(-2, -3)
