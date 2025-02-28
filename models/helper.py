import torch.nn as nn
import inspect

from models.heads import HEADS_REGISTRY
from models.transformers import TRANSFORMERS_REGISTRY


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
    model = nn.Sequential(transformer, head)
    return model, transformer_defaults, head_defaults
