from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY


def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        feedforward_norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        attention_activation_layer: Type[nn.Module] = nn.GELU,
        feedforward_activation_layer: Type[nn.Module] = nn.GELU,
        attention_linear_layer: Type[nn.Linear] = nn.Linear,
        feedforward_linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ViTAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            norm_layer=attention_norm_layer,
                            activation_layer=attention_activation_layer,
                            linear_layer=attention_linear_layer,
                        ),
                        ViTFeedForward(
                            dim,
                            mlp_dim,
                            dropout=dropout,
                            norm_layer=feedforward_norm_layer,
                            activation_layer=feedforward_activation_layer,
                            linear_layer=feedforward_linear_layer,
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
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        in_channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0,
        embedding_norm: str = "LayerNorm",
        embedding_linear: str = "Linear",
        attention_linear_layer: str = "Linear",
        attention_norm_layer: str = "LayerNorm",
        feedforward_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        feedforward_activation_layer: str = "GELU",
    ) -> None:
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_norm_layer=LAYERS_REGISTRY[attention_norm_layer.lower()],
            feedforward_norm_layer=LAYERS_REGISTRY[feedforward_norm_layer.lower()],
            attention_activation_layer=LAYERS_REGISTRY[
                attention_activation_layer.lower()
            ],
            feedforward_activation_layer=LAYERS_REGISTRY[
                feedforward_activation_layer.lower()
            ],
            attention_linear_layer=LAYERS_REGISTRY[attention_linear_layer.lower()],
            feedforward_linear_layer=LAYERS_REGISTRY[feedforward_linear_layer.lower()],
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


class ViTSmall(ViT):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        dim: int = 384,  # smaller embedding dimension
        depth: int = 8,  # fewer transformer layers
        heads: int = 6,  # fewer attention heads
        mlp_dim: int = 1536,  # smaller MLP dimension
        in_channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0,
        embedding_norm: str = "LayerNorm",
        embedding_linear: str = "Linear",
        attention_linear_layer: str = "Linear",
        attention_norm_layer: str = "LayerNorm",
        feedforward_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        feedforward_activation_layer: str = "GELU",
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            embedding_norm=embedding_norm,
            embedding_linear=embedding_linear,
            attention_linear_layer=attention_linear_layer,
            attention_norm_layer=attention_norm_layer,
            feedforward_linear_layer=feedforward_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            attention_activation_layer=attention_activation_layer,
            feedforward_activation_layer=feedforward_activation_layer,
        )
    


class MiniViT(ViT):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 4,
        dim: int = 128,  # smaller embedding dimension
        depth: int = 2,  # fewer transformer layers
        heads: int = 4,  # fewer attention heads
        mlp_dim: int = 128 * 3,  # smaller MLP dimension
        in_channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0,
        embedding_norm: str = "LayerNorm",
        embedding_linear: str = "Linear",
        attention_linear_layer: str = "Linear",
        attention_norm_layer: str = "LayerNorm",
        feedforward_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        feedforward_activation_layer: str = "GELU",
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            embedding_norm=embedding_norm,
            embedding_linear=embedding_linear,
            attention_linear_layer=attention_linear_layer,
            attention_norm_layer=attention_norm_layer,
            feedforward_linear_layer=feedforward_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            attention_activation_layer=attention_activation_layer,
            feedforward_activation_layer=feedforward_activation_layer,
        )
    
