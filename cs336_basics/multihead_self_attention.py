import torch
import torch.nn as nn

from einops import rearrange

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    """
    Applies multi-head self-attention to the incoming data.

    This module performs linear projections for queries, keys, and values,
    splits them into multiple heads, applies scaled dot-product attention
    with a causal mask, and projects the concatenated outputs back to the
    original model dimension.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        if theta and max_seq_len:
            self.rope = RotaryPositionalEmbedding(d_model // num_heads, theta, max_seq_len, device=device)
        else:
            self.rope = None
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculates the forward pass of the multi-head self-attention.

        Args:
            x (torch.Tensor): The input tensor of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (..., seq_len, d_model)
                         after applying multi-head self-attention.
        """

        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        q_proj = rearrange(
            q_proj, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )
        k_proj = rearrange(
            k_proj, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )
        v_proj = rearrange(
            v_proj, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )

        if self.rope:
            q_proj = self.rope(q_proj, token_positions)
            k_proj = self.rope(k_proj, token_positions)

        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attention = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask=mask)
        attention = rearrange(attention, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        return self.o_proj(attention)
