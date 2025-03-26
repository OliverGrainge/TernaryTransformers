import torch
import torch.nn as nn
import inspect

from models.heads import HEADS_REGISTRY
from models.transformers import TRANSFORMERS_REGISTRY
from config import ModelConfig, BackboneConfig, HeadConfig


class CustomModel(nn.Module):
    def __init__(self, transformer, head):
        super().__init__()
        self.transformer = transformer
        self.head = head

    def forward(self, x, **kwargs):
        transformer_output = self.transformer(x, **kwargs)
        return self.head(transformer_output)

    def compute_decay(self, reduction: str = "mean") -> torch.Tensor:
        if reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")

        if not hasattr(self, "_decay_modules"):
            self._decay_modules = [
                m for m in self.modules() if hasattr(m, "compute_layer_decay")
            ]

        decay_loss = torch.tensor(0.0).to(next(self.parameters()).device)
        n_modules = 0

        for layer in self._decay_modules:
            decay = layer.compute_layer_decay()
            if decay is not None:
                n_modules += 1
                decay_loss += decay

        if reduction == "mean":
            return decay_loss / max(n_modules, 1)  # Avoid division by zero
        return decay_loss  # sum case

    def set_progress(self, progress: float):
        if not hasattr(self, "_progress_modules"):
            self._progress_modules = [
                m for m in self.modules() if hasattr(m, "set_layer_progress")
            ]

        for layer in self._progress_modules:
            layer.weight.data = layer.weight.data * progress


def create_model(
    model_config: ModelConfig,
):
    transformer_cls = TRANSFORMERS_REGISTRY[model_config.backbone.backbone.lower()]
    head_cls = HEADS_REGISTRY[model_config.head.head.lower()]

    # Create model components with all parameters
    transformer = transformer_cls(model_config.backbone)
    head = head_cls(model_config.head)
    model = CustomModel(transformer, head)
    return model
