import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    A mathematically simplified and computationally cheaper alternative to LayerNorm
    that normalizes activations strictly by their root mean square, ignoring mean centering.
    Widely used in modern architectures like LLaMA.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.
        Calculations are internally upcasted to float32 for numerical stability
        before being scaled by the learnable weight parameter.
        Args:
            x (torch.Tensor): The input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: The normalized tensor of shape (..., d_model), returned
                          in the original input data type.
        """
        in_type = x.dtype
        x = x.to(torch.float)
        square_mean = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(square_mean + self.eps)
        x_normed = x / rms
        result = x_normed * self.weight
        return result.to(in_type)
