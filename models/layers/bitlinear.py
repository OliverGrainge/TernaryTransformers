from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# More can be found about this layer here:https://arxiv.org/abs/2402.17764


class WeightQuantizer(Function):
    @staticmethod
    def forward(ctx, w):
        delta = w.abs().mean().clamp(min=1e-5)
        scale = 1.0 / delta
        qw = (w * scale).round().clamp_(-1, 1)
        dqw = delta * qw
        ctx.save_for_backward(scale)
        return dqw, scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        (scale,) = ctx.saved_tensors
        return grad_output * scale


class ActivationQuantizer(Function):
    @staticmethod
    def forward(ctx, x):
        scale = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
        qx = (x * scale).round().clamp_(-128, 127)
        ctx.save_for_backward(scale)
        return qx, scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        (scale,) = ctx.saved_tensors
        return grad_output * scale


def quant_weight(w: torch.Tensor):
    return WeightQuantizer.apply(w)


def quant_activation(x: torch.Tensor):
    return ActivationQuantizer.apply(x)


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        # RMS normalization applied to the input before quantization
        self.norm = nn.RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMS normalization to the input
        x = self.norm(x)

        # Quantize weights and activations
        quantized_weight, weight_scale = quant_weight(self.weight)
        quantized_input, input_scale = quant_activation(x)

        # Compute the linear transformation using the quantized values
        output = F.linear(quantized_input, quantized_weight)

        # Rescale the output using the computed scaling factors
        expand_dims = (1,) * (x.ndim - 1)  # For proper broadcasting over dimensions
        scaled_output = output / (
            input_scale.view(-1, *expand_dims) * weight_scale.view(*expand_dims, -1)
        )

        # Add bias if present
        if self.bias is not None:
            scaled_output = scaled_output + self.bias

        return scaled_output
