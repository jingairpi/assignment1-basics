import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Computes the learning rate at a given iteration under a cosine decay schedule with linear warmup.

    The schedule increases the learning rate linearly from 0 to max_learning_rate during
    the warmup iterations, then decays it to min_learning_rate following a cosine curve up
    to cosine_cycle_iters. For any iterations after cosine_cycle_iters, the learning rate
    remains constant at min_learning_rate.

    Args:
        it (int): The current iteration number.
        max_learning_rate (float): The maximum learning rate at the peak of warmup.
        min_learning_rate (float): The minimum/final learning rate after cosine decay.
        warmup_iters (int): The number of warmup iterations.
        cosine_cycle_iters (int): The total number of iterations for the warmup + decay cycle.

    Returns:
        float: The learning rate at iteration `it`.
    """

    assert warmup_iters < cosine_cycle_iters

    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters

    if it <= cosine_cycle_iters:
        return (
            min_learning_rate
            + (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)))
            * (max_learning_rate - min_learning_rate)
            / 2
        )

    return min_learning_rate
