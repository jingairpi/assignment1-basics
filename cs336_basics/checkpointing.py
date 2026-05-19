import os
import torch
import torch.nn as nn

from typing import IO, BinaryIO


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save a training checkpoint containing the model state, optimizer state, and current iteration.

    Args:
        model: The PyTorch module whose parameters should be saved.
        optimizer: The PyTorch optimizer whose state should be saved.
        iteration: The current training iteration index.
        out: The destination path (string or path-like object) or a writable binary
            file-like object where the checkpoint will be saved.
    """
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["iteration"] = iteration
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """
    Load and restore a training checkpoint, returning the saved iteration number.

    Args:
        src: The source path (string or path-like object) or a readable binary
            file-like object containing the serialized checkpoint.
        model: The PyTorch module to restore the parameters to.
        optimizer: The PyTorch optimizer to restore the state to.

    Returns:
        The iteration index at which the checkpoint was saved.
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    state = checkpoint.get("model", None)
    if state is None:
        raise KeyError(f"Model is not found from {src}")
    model.load_state_dict(state)
    state = checkpoint.get("optimizer", None)
    if state is None:
        raise KeyError(f"Optimizer is not found from {src}")
    optimizer.load_state_dict(state)
    if "iteration" not in checkpoint:
        raise KeyError(f"Iteration is not found from {src}")
    return checkpoint["iteration"]
