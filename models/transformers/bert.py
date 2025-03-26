from typing import Tuple, Type, Union, Optional

import torch
import torch.nn as nn
from einops import repeat

# from einops.layers.torch import Rearrange  # not really needed for BERT embeddings

from models.blocks import ViTAttention, ViTFeedForward, TransformerAttention
from models.layers import LAYERS_REGISTRY
from config import BackboneConfig

class Transformer(nn.Module):
    """
    Reuse your existing Transformer block exactly as you provided.
    (ViTAttention + ViTFeedForward repeated 'depth' times)
    """

    def __init__(
        self,
        backbone_config: BackboneConfig,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(backbone_config.dim)
        self.layers = nn.ModuleList([])
        for _ in range(backbone_config.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(
                            backbone_config.dim,
                            heads=backbone_config.heads,
                            dim_head=backbone_config.dim_head,
                            dropout=backbone_config.dropout,
                            norm_layer=LAYERS_REGISTRY[backbone_config.attention_norm_layer],
                            activation_layer=LAYERS_REGISTRY[backbone_config.attention_activation_layer],
                            linear_layer=LAYERS_REGISTRY[backbone_config.attention_linear_layer],
                        ),
                        ViTFeedForward(
                            backbone_config.dim,
                            backbone_config.ffn_dim,
                            dropout=backbone_config.dropout,
                            norm_layer=LAYERS_REGISTRY[backbone_config.feedforward_norm_layer],
                            activation_layer=LAYERS_REGISTRY[backbone_config.feedforward_activation_layer],
                            linear_layer=LAYERS_REGISTRY[backbone_config.feedforward_linear_layer],
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
      2) Passes the embedded sequence through a Transformer (ViTAttention + ViTFeedForward).
      3) Returns a final sequence of hidden states [batch_size, seq_len, dim].
    """

    def __init__(
        self,
        backbone_config: BackboneConfig,
    ) -> None:
        super().__init__()

        # --- Embedding Layers ---
        self.token_embedding = nn.Embedding(
            num_embeddings=backbone_config.vocab_size, embedding_dim=backbone_config.dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=backbone_config.max_seq_len, embedding_dim=backbone_config.dim
        )
        self.segment_embedding = nn.Embedding(
            num_embeddings=backbone_config.num_segments, embedding_dim=backbone_config.dim
        )

        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(backbone_config.emb_dropout)

        # --- Transformer ---
        self.transformer = Transformer(
            dim=backbone_config.dim,
            depth=backbone_config.depth,
            heads=backbone_config.heads,
            dim_head=backbone_config.dim_head,
            mlp_dim=backbone_config.mlp_dim,
            dropout=backbone_config.dropout,
            attention_norm_layer=LAYERS_REGISTRY[backbone_config.attention_norm_layer.lower()],
            feedforward_norm_layer=LAYERS_REGISTRY[backbone_config.feedforward_norm_layer.lower()],
            attention_activation_layer=LAYERS_REGISTRY[
                backbone_config.attention_activation_layer.lower()
            ],
            feedforward_activation_layer=LAYERS_REGISTRY[
                backbone_config.feedforward_activation_layer.lower()
            ],
            attention_linear_layer=LAYERS_REGISTRY[backbone_config.attention_linear_layer.lower()],
            feedforward_linear_layer=LAYERS_REGISTRY[backbone_config.feedforward_linear_layer.lower()],
        )

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
        return x
