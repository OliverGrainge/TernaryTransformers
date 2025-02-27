import torch
from torch.autograd import Function

"""
Quantization Function Naming Convention

All quantization functions in this module follow a standardized naming scheme to ensure clarity, consistency, 
and scalability when adding new quantization methods. The naming pattern is defined as:

    quantize_<TARGET>_<PRECISION>_<SCALING>_<GRANULARITY>_<MODE>_<BACKWARD>

where:

    <TARGET>:
        - 'W': Quantization for weights.
        - 'A': Quantization for activations.

    <PRECISION>:
        - e.g., 'I8', 'I4' for integer quantization.
        - 'FP' for floating point.
        - 'TERNARY' for ternary quantization.

    <SCALING>:
        - 'S': Symmetric scaling.
        - 'A': Asymmetric scaling.

    <GRANULARITY>:
        - 'PT': Per-tensor quantization.
        - 'CH': Per-channel quantization.
        - 'TG': Tensor-group quantization.

    <MODE>:
        - 'dyn': Dynamic quantization (parameters computed on the fly).
        - 'stat': Static quantization (pre-calibrated parameters).

    <BACKWARD>:
        - 'STE': Using the Straight-Through Estimator for the backward pass.
        - (Additional tokens may be added here for alternative backward methods.)

Examples:
    - quantize_W_I8_S_CH_stat_STE:
        -> Quantizes weights using 8-bit integers, symmetric scaling, per-channel granularity,
           static quantization, and the STE for the backward pass.

    - quantize_A_I8_S_PT_dyn_STE:
        -> Quantizes activations using 8-bit integers, symmetric scaling, per-tensor granularity,
           dynamic quantization, and the STE for the backward pass.

This naming convention is inspired by best practices in quantized neural network research 
(e.g., Courbariaux et al., 2015; Jacob et al., 2018) and is designed to make the implementation 
self-documenting and easily extendable.
"""


class Quantizer_A_I8_S_PT_dyn_STE(Function):
    @staticmethod
    def forward(ctx, x, outtype=None):
        if outtype is None:
            outtype = x.type()
        scale = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
        qx = (x * scale).round().clamp_(-128, 127).type(outtype)
        ctx.save_for_backward(scale)
        return qx, scale.flatten(), None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        (scale,) = ctx.saved_tensors
        # STE: Pass gradients straight-through, accounting for scaling
        return grad_output * scale, None


def quant_A_I8_S_PT_dyn_STE(x: torch.Tensor, outtype=None):
    return Quantizer_A_I8_S_PT_dyn_STE.apply(x, outtype)
