import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    A Transformer-based Language Model.

    This module combines token embeddings, a stack of Transformer blocks,
    a final layer normalization, and a linear output head to predict the
    scores (logits) for the next token in the sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the Transformer language model.

        Args:
            x (torch.Tensor): The input tensor of integer token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, seq_len, vocab_size).
        """

        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
