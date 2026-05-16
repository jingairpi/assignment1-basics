import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the numerically stable softmax of the input tensor along a specified dimension.

    This implementation subtracts the maximum value along the dimension before
    exponentiation to prevent numerical overflow.

    Args:
        x: The input tensor.
        dim: The dimension along which to compute the softmax.

    Returns:
        A tensor of the same shape as x, where the values along the specified
        dimension sum to 1.
    """

    x_max = torch.amax(x, dim, keepdim=True)
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    return x_exp / torch.sum(x_exp, dim, keepdim=True)
