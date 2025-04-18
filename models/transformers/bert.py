from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from models.blocks import FeedForward, TransformerAttention, ViTAttention
from models.layers import LAYERS_REGISTRY

# from einops.layers.torch import Rearrange  # not really needed for BERT embeddings


class Transformer(nn.Module):
    """
    Transformer block with self-attention and feedforward layers
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        attention_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        feedforward_activation_layer: str = "GELU",
        feedforward_linear_layer: str = "Linear",
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            norm_layer=LAYERS_REGISTRY[attention_norm_layer.lower()],
                            activation_layer=LAYERS_REGISTRY[attention_activation_layer.lower()],
                            linear_layer=LAYERS_REGISTRY[attention_linear_layer.lower()],
                        ),
                        FeedForward(
                            dim,
                            ffn_dim,
                            dropout=dropout,
                            norm_layer=LAYERS_REGISTRY[feedforward_norm_layer.lower()],
                            activation_layer=LAYERS_REGISTRY[feedforward_activation_layer.lower()],
                            linear_layer=LAYERS_REGISTRY[feedforward_linear_layer.lower()],
                        ),
                    ]
                )
            )

    

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, attention_mask=attention_mask) + x
            x = ff(x) + x
        return self.norm(x)


class Bert(nn.Module):
    """
    Minimal BERT-style model that:
      1) Creates token embeddings, positional embeddings, and (optional) segment embeddings.
      2) Passes the embedded sequence through a Transformer (TransformerAttention + FeedForward).
      3) Returns a final sequence of hidden states [batch_size, seq_len, dim].
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_segments: int = 2,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = None,
        ffn_dim: int = None,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        attention_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        attention_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        feedforward_activation_layer: str = "GELU",
        feedforward_linear_layer: str = "Linear",
    ) -> None:
        """
        Args:
            vocab_size (int): Size of vocabulary
            context_length (int): Maximum sequence length
            num_segments (int, optional): Number of segment types. Defaults to 2
            dim (int, optional): Model dimension. Defaults to 768
            depth (int, optional): Number of transformer layers. Defaults to 12
            heads (int, optional): Number of attention heads. Defaults to 12
            dim_head (int, optional): Dimension of each attention head. Defaults to dim // heads
            ffn_dim (int, optional): Feedforward network dimension. Defaults to 4 * dim
            dropout (float, optional): Dropout rate. Defaults to 0.1
            embedding_dropout (float, optional): Embedding dropout rate. Defaults to 0.1
            attention_norm_layer (str, optional): Normalization layer for attention. Defaults to "LayerNorm"
            attention_activation_layer (str, optional): Activation layer for attention. Defaults to "GELU"
            attention_linear_layer (str, optional): Linear layer for attention. Defaults to "Linear"
            feedforward_norm_layer (str, optional): Normalization layer for feedforward. Defaults to "LayerNorm"
            feedforward_activation_layer (str, optional): Activation layer for feedforward. Defaults to "GELU"
            feedforward_linear_layer (str, optional): Linear layer for feedforward. Defaults to "Linear"
        """
        super().__init__()

        # Set default values for dim_head and ffn_dim if not provided
        dim_head = dim_head if dim_head is not None else dim // heads
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * dim

        # --- Embedding Layers ---
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=dim,
        )
        self.segment_embedding = nn.Embedding(
            num_embeddings=num_segments,
            embedding_dim=dim,
        )

        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # --- Transformer ---
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_norm_layer=attention_norm_layer,
            attention_activation_layer=attention_activation_layer,
            attention_linear_layer=attention_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            feedforward_activation_layer=feedforward_activation_layer,
            feedforward_linear_layer=feedforward_linear_layer,
        )

        self.logits = nn.Linear(dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            attention_mask: [batch_size, seq_len] - 1 for tokens to attend to, 0 for padding tokens
            token_type_ids: [batch_size, seq_len] - segment IDs (optional)
        Returns:
            Hidden states of shape [batch_size, seq_len, dim].
        """
        bsz, seq_len = input_ids.shape

        # Create position indices (0,1,2,...,seq_len-1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(bsz, seq_len)

        if attention_mask is None:
            # if no attention mask provided, default to attending to all tokens
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        if token_type_ids is None:
            # if no segment IDs provided, default to zeros
            token_type_ids = torch.zeros_like(
                input_ids, dtype=torch.long, device=input_ids.device
            )

        # Sum of token, position, and segment embeddings
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(token_type_ids)
        )

        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Pass through the Transformer with attention mask
        x = self.transformer(x, attention_mask=attention_mask)
        logits = self.logits(x)
        return logits
