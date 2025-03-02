from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BaseQuantizerSTE(Function):
    """
    Base class for quantizers using the Straight Through Estimator (STE).
    
    The subclass should provide a scaling function and clamping bounds.
    """
    @staticmethod
    def forward(ctx, x, outtype, scale_fn, clamp_min: int, clamp_max: int):
        # Compute scaling factor using provided scale_fn
        scale = scale_fn(x)
        # Quantize: scale, round, clamp, and cast to outtype
        qx = (x * scale).round().clamp_(clamp_min, clamp_max).type(outtype)
        ctx.save_for_backward(scale)
        return qx, scale.flatten(), None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        (scale,) = ctx.saved_tensors
        # STE: Pass the gradient through, scaling appropriately
        return grad_output * scale, None


class Quantizer_A_I8_S_PT_dyn_STE(BaseQuantizerSTE):
    """
    Activation quantizer: maps inputs to 8-bit signed integers with dynamic scaling.
    
    Scale is computed per-row:
      scale = 127.0 / max(|x|, 1e-5)
    with clamping to the range [-128, 127].
    """
    @staticmethod
    def forward(ctx, x, outtype=None):
        if outtype is None:
            outtype = x.type()
        scale_fn = lambda x: 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
        return BaseQuantizerSTE.forward(ctx, x, outtype, scale_fn, -128, 127)


def quant_A_I8_S_PT_dyn_STE(x: torch.Tensor, outtype=None):
    return Quantizer_A_I8_S_PT_dyn_STE.apply(x, outtype)


class Quantizer_W_TERNARY_S_PT_dyn_STE(BaseQuantizerSTE):
    """
    Weight quantizer: maps weights to a ternary representation with dynamic scaling.
    
    Scale is computed over the entire weight tensor:
      scale = 1.0 / mean(|w|, 1e-5)
    with clamping to the range [-1, 1].
    """
    @staticmethod
    def forward(ctx, w, outtype=None):
        if outtype is None:
            outtype = w.type()
        scale_fn = lambda w: 1.0 / w.abs().mean().clamp(min=1e-5)
        return BaseQuantizerSTE.forward(ctx, w, outtype, scale_fn, -1, 1)


def quant_W_TERNARY_S_PT_dyn_STE(w: torch.Tensor, outtype=None):
    return Quantizer_W_TERNARY_S_PT_dyn_STE.apply(w, outtype)




class BitLinear(nn.Linear):
    """
    Combined quantized linear layer with symmetric quantization and input RMS normalization.

    This layer performs the following steps:
      1. Applies RMS normalization to the input.
      2. Quantizes the input activations and weights using the provided quantizers:
         - Activations: quant_A_I8_S_PT_dyn_STE (8-bit signed dynamic quantization).
         - Weights: quant_W_TERNARY_S_PT_dyn_STE (ternary quantization).
      3. Performs a linear transformation using the quantized values.
      4. Rescales the output using the computed scaling factors.
      5. Optionally adds a bias.

    The design maintains the functionality of the original BitLinear class.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        # Assign quantization functions for weights and activations
        self.weight_quant_fn = quant_W_TERNARY_S_PT_dyn_STE
        self.activation_quant_fn = quant_A_I8_S_PT_dyn_STE
        # RMS normalization applied to the input before quantization
        self.norm = nn.RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMS normalization to the input
        x = self.norm(x)

        # Quantize weights and activations
        quantized_weight, weight_scale, _ = self.weight_quant_fn(self.weight, outtype=None)
        quantized_input, input_scale, _ = self.activation_quant_fn(x, outtype=None)

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
    
