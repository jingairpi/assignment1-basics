import torch
import torch.nn as nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.multihead_self_attention import MultiheadSelfAttention


class TransformerBlock(nn.Module):
    """
    A single layer of a Transformer model using the Pre-LN architecture.

    This module combines Multi-Head Self-Attention and a Position-Wise
    Feed-Forward network (SwiGLU). It applies RMSNorm before each
    sub-layer and includes residual connections after each sub-layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the Transformer block.

        Args:
            x (torch.Tensor): The input hidden states of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: The output hidden states of shape (..., seq_len, d_model)
                         after applying attention and feed-forward operations.
        """

        token_positions = torch.arange(x.size(-2), device=x.device)
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        return y + self.ffn(self.ln2(y))
