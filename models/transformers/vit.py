from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY
from config import BackboneConfig

def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


class Transformer(nn.Module):
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
                        ViTAttention(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        backbone_config: BackboneConfig,
    ) -> None:
        super().__init__()

        image_height, image_width = pair(backbone_config.image_size)
        patch_height, patch_width = pair(backbone_config.patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = backbone_config.in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, backbone_config.dim),
            nn.LayerNorm(backbone_config.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, backbone_config.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, backbone_config.dim))
        self.dropout = nn.Dropout(backbone_config.emb_dropout)

        self.transformer = Transformer(
                backbone_config=backbone_config,
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

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x






