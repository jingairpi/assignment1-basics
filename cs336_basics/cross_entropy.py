import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss across examples.

    This function implements the Log-Sum-Exp trick for numerical stability,
    preventing overflow when exponentiating large logits.

    Args:
        inputs (torch.Tensor): The unnormalized logits of shape (batch_size, vocab_size).
        targets (torch.Tensor): The ground truth class indices of shape (batch_size,).

    Returns:
        torch.Tensor: A scalar tensor containing the average cross-entropy loss.
    """

    batch_indices = torch.arange(targets.size(0))
    target_logits = inputs[batch_indices, targets]
    max_vals = inputs.max(dim=1, keepdim=True).values
    exp_vals = torch.exp(inputs - max_vals)
    sum_exp_vals = exp_vals.sum(dim=1)
    log_sum_exp_vals = torch.log(sum_exp_vals) + max_vals.squeeze(1)
    return torch.mean(log_sum_exp_vals - target_logits)
