import torch
import torch.nn as nn


class RMSNorm(nn.Module):
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
        in_type = x.dtype
        x = x.to(torch.float)
        square_mean = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(square_mean + self.eps)
        x_normed = x / rms
        result = x_normed * self.weight
        return result.to(in_type)
