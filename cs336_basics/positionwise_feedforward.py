import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    """
    A position-wise feed-forward network utilizing the SwiGLU activation mechanism.

    Combines a Swish (SiLU) activation function with a Gated Linear Unit (GLU).
    The input is projected independently via two linear layers; one pathway is
    activated and acts as a multiplicative gate to the other, before a final
    linear down-projection.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the SwiGLU forward pass over the sequence.
        Args:
            x (torch.Tensor): The input hidden states of shape (..., d_model).

        Returns:
            torch.Tensor: The output hidden states of shape (..., d_model).
        """
        w1_out = self.w1(x)
        w3_out = self.w3(x)

        hidden = F.silu(w1_out) * w3_out

        return self.w2(hidden)
