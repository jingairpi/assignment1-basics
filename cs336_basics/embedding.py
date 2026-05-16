import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module retrieves dense continuous vectors given discrete integer token IDs,
    acting as the first layer mapping tokens into the model's hidden dimension.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the embedding vectors for a batch of token IDs.
        Args:
            token_ids (torch.Tensor): A tensor of integer token IDs of shape (...,).
                                      Often shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The mapped dense embeddings of shape (..., embedding_dim).
        """
        return self.weight[token_ids]
