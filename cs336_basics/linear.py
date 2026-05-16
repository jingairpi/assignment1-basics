import math
import torch
import torch.nn as nn

from einops import einsum


class Linear(nn.Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T.

    This module performs a learnable matrix multiplication without a bias term,
    matching modern LLM architectural standards.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(2 / (in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the linear projection.
        Args:
            x (torch.Tensor): The input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: The projected output tensor of shape (..., out_features).
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
