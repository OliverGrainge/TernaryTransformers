import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from models.quantization import *


class QLinearSymmetric(nn.Linear):
    """Quantized linear layer with scaling factors.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds a learnable bias to the output
        weight_quant_fn: Function to quantize weights
        activation_quant_fn: Function to quantize activations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_quant_fn: Callable[
            [torch.Tensor, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        activation_quant_fn: Callable[
            [torch.Tensor, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.weight_quant_fn = weight_quant_fn
        self.activation_quant_fn = activation_quant_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and inputs
        quantized_weight, weight_scale, weight_zero = self.weight_quant_fn(
            self.weight, outtype=None
        )
        quantized_input, input_scale, input_zero = self.activation_quant_fn(
            x, outtype=None
        )

        # Compute linear transformation
        output = F.linear(quantized_input, quantized_weight)

        # Apply scaling factors
        expand_dims = (1,) * (x.ndim - 1)  # Create tuple of 1s for proper broadcasting
        scaled_output = output / (
            input_scale.view(-1, *expand_dims) * weight_scale.view(*expand_dims, -1)
        )

        # Add bias if present"?|""
        if self.bias is not None:
            scaled_output = scaled_output + self.bias

        return scaled_output


class BitLinear(QLinearSymmetric):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            weight_quant_fn=quant_W_TERNARY_S_PT_dyn_STE,
            activation_quant_fn=quant_A_I8_S_PT_dyn_STE,
        )

        self.norm = nn.RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = super().forward(x)
        return x

