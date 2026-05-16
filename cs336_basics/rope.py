import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Encodes positional information directly into the queries and keys by rotating
    their representations in the complex plane. RoPE applies a relative positional
    encoding using precomputed trigonometric frequencies, allowing the model to
    better generalize to sequence lengths beyond the training distribution.
    """

    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device | str | None = None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k // 2, device=device)
        angles = 1 / theta ** (2 * k / d_k)
        positions = torch.arange(max_seq_len, device=device)
        rotations = torch.outer(positions, angles)

        cos_tensor = torch.cos(rotations)
        sin_tensor = torch.sin(rotations)

        self.register_buffer("cos_cached", cos_tensor)
        self.register_buffer("sin_cached", sin_tensor)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional embeddings to the input tensor.
        Extracts the precomputed cosine and sine frequencies corresponding to the
        provided token positions, and applies a 2D rotation matrix to interleaved
        pairs of feature dimensions.
        Args:
            x (torch.Tensor): The input tensor (queries or keys) to be rotated.
                              Typically of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Integer tensor of sequence positions
                                            corresponding to the inputs.
                                            Shape must broadcast against x.
        Returns:
            torch.Tensor: The rotated tensor of the exact same shape as the input `x`.
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1, x2 = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.stack([y1, y2], dim=-1).flatten(-2)
