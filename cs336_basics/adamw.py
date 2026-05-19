import math
import torch
import torch.nn as nn

from collections.abc import Callable, Iterable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW optimization algorithm.

    AdamW is a variant of Adam that decouples weight decay from the gradient
    updates, leading to more effective regularization.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Initializes the AdamW optimizer.

        Args:
            params (Iterable[nn.Parameter]): An iterable of parameters to optimize.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay coefficient. Defaults to 0.01.
            betas (tuple[float, float], optional): Coefficients used for computing running
                averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical
                stability. Defaults to 1e-8.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model
                and returns the loss.

        Returns:
            torch.Tensor | None: The loss if closure is provided, else None.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p, device=p.device))
                v = state.get("v", torch.zeros_like(p, device=p.device))

                step_size = lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                with torch.no_grad():
                    p.mul_(1 - lr * weight_decay)
                    m = betas[0] * m + (1 - betas[0]) * p.grad
                    v = betas[1] * v + (1 - betas[1]) * p.grad**2
                    p.add_(m / (torch.sqrt(v) + eps), alpha=-step_size)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
