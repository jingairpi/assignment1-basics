import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
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
        w1_out = self.w1(x)
        w3_out = self.w3(x)

        hidden = F.silu(w1_out) * w3_out

        return self.w2(hidden)
