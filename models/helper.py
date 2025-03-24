import torch
import torch.nn as nn
import inspect

from models.heads import HEADS_REGISTRY
from models.transformers import TRANSFORMERS_REGISTRY


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
    backbone: str, head: str, backbone_kwargs: dict = {}, head_kwargs: dict = {}
):
    transformer_cls = TRANSFORMERS_REGISTRY[backbone.lower()]
    head_cls = HEADS_REGISTRY[head.lower()]

    # Get default arguments for both classes
    transformer_params = inspect.signature(transformer_cls).parameters
    head_params = inspect.signature(head_cls).parameters

    # Create dictionaries with default values
    transformer_defaults = {
        name: param.default
        for name, param in transformer_params.items()
        if param.default is not inspect.Parameter.empty
    }
    head_defaults = {
        name: param.default
        for name, param in head_params.items()
        if param.default is not inspect.Parameter.empty
    }

    # Update defaults with provided kwargs
    transformer_defaults.update(backbone_kwargs)
    head_defaults.update(head_kwargs)

    # Create model components with all parameters
    transformer = transformer_cls(**transformer_defaults)
    head = head_cls(**head_defaults)
    model = CustomModel(transformer, head)

    return model, transformer_defaults, head_defaults
