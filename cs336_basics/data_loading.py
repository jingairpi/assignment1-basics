import torch

import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a 1D dataset of token IDs, sample a batch of input sequences and their
    corresponding next-token target labels.

    Args:
        dataset: A 1D NumPy array containing the token IDs of the dataset.
        batch_size: The number of sequences to sample in the batch.
        context_length: The length of each input sequence.
        device: The PyTorch device (e.g., 'cpu' or 'cuda') to place the tensors on.

    Returns:
        A tuple of (x, y) where x is the input sequences of shape (batch_size, context_length)
        and y is the target sequences of shape (batch_size, context_length).
    """
    dataset_tensor = torch.from_numpy(dataset)
    size = len(dataset_tensor)
    max_index = size - context_length - 1
    start_indices = torch.randint(0, max_index + 1, (batch_size,))
    start_indices = start_indices[:, None]
    offset = torch.arange(0, context_length)
    indices = start_indices + offset
    x = dataset_tensor[indices].to(device)
    y = dataset_tensor[indices + 1].to(device)
    return (x, y)
