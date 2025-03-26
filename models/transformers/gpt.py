from typing import Tuple, Type, Union, Optional

import torch
import torch.nn as nn
from einops import repeat

from models.blocks import ViTAttention, ViTFeedForward, TransformerAttention
from models.layers import LAYERS_REGISTRY
from config import BackboneConfig


class CausalTransformer(nn.Module):
    """
    Similar to the Transformer block but with causal attention masking
    for autoregressive prediction.
    """

    def __init__(
        self,
        backbone_config: BackboneConfig,
    ) -> None:
        super().__init__()
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
                            is_causal=True,  # This is the key difference from BERT
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
        self.norm = nn.LayerNorm(backbone_config.dim)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, attention_mask=attention_mask) + x
            x = ff(x) + x
        return self.norm(x)


class GPT(nn.Module):
    """
    Minimal GPT-style model that:
      1) Creates token embeddings and positional embeddings
      2) Passes the embedded sequence through a causal Transformer
      3) Returns logits for next token prediction [batch_size, seq_len, vocab_size]
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

        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(backbone_config.emb_dropout)

        # --- Transformer ---
        self.transformer = CausalTransformer(
            backbone_config,
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
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            attention_mask: [batch_size, seq_len] - 1 for tokens to attend to, 0 for padding tokens
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size] for next token prediction
        """
        bsz, seq_len = input_ids.shape

        # Create position indices (0,1,2,...,seq_len-1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(bsz, seq_len)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        # Sum of token and position embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Pass through the causal Transformer
        x = self.transformer(x, attention_mask=attention_mask)
        return x
