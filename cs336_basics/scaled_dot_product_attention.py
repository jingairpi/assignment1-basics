import math
import torch

from einops import einsum
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.
    Args:
        q: Query tensor of shape (..., seq_q, d_k)
        k: Key tensor of shape (..., seq_k, d_k)
        v: Value tensor of shape (..., seq_k, d_v)
        mask: Optional boolean mask tensor of shape (..., seq_q, seq_k).
              True indicates values to keep, False indicates values to mask.
    Returns:
        The output tensor of shape (..., seq_q, d_v).
    """
    d_k = q.size(-1)
    scores = einsum(q, k, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")

    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))

    attn_probs = softmax(scores / math.sqrt(d_k), -1)
    return einsum(attn_probs, v, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
