import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.amax(x, dim, keepdim=True)
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    return x_exp / torch.sum(x_exp, dim, keepdim=True)
