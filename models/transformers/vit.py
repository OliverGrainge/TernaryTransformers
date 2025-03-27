from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from config import ModelConfig
from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY


def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


class Transformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_config.transformer_dim)
        self.layers = nn.ModuleList([])
        for _ in range(model_config.transformer_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ViTAttention(
                            model_config.transformer_dim,
                            heads=model_config.transformer_heads,
                            dim_head=model_config.transformer_dim_head,
                            dropout=model_config.transformer_dropout,
                            norm_layer=LAYERS_REGISTRY[
                                model_config.attention_norm_layer.lower()
                            ],
                            activation_layer=LAYERS_REGISTRY[
                                model_config.attention_activation_layer.lower()
                            ],
                            linear_layer=LAYERS_REGISTRY[
                                model_config.attention_linear_layer.lower()
                            ],
                        ),
                        ViTFeedForward(
                            model_config.transformer_dim,
                            model_config.transformer_ffn_dim,
                            dropout=model_config.transformer_dropout,
                            norm_layer=LAYERS_REGISTRY[
                                model_config.feedforward_norm_layer.lower()
                            ],
                            activation_layer=LAYERS_REGISTRY[
                                model_config.feedforward_activation_layer.lower()
                            ],
                            linear_layer=LAYERS_REGISTRY[
                                model_config.feedforward_linear_layer.lower()
                            ],
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()

        image_height, image_width = pair(model_config.image_size)
        patch_height, patch_width = pair(model_config.image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = model_config.image_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_config.transformer_dim),
            nn.LayerNorm(model_config.transformer_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, model_config.transformer_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_config.transformer_dim))
        self.dropout = nn.Dropout(model_config.embedding_dropout)

        self.transformer = Transformer(model_config=model_config)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
