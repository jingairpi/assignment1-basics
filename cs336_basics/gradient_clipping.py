import torch
import torch.nn as nn

from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips the combined gradients of parameters to have an L2 norm of at most max_l2_norm.

    The gradients of the parameters are modified in-place.

    Args:
        parameters (Iterable[nn.Parameter]): An iterable of parameters whose gradients
            will be clipped.
        max_l2_norm (float): The maximum allowed L2 norm of the combined gradients.
    """

    total_sq = 0
    for param in parameters:
        if param.grad is not None:
            total_sq += param.grad.detach().pow(2).sum()

    total_norm = torch.sqrt(total_sq)
    if total_norm < max_l2_norm:
        return

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    for param in parameters:
        if param.grad is not None:
            with torch.no_grad():
                param.grad.mul_(clip_coef)
