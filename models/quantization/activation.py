import torch 
from torch.autograd import Function

# QA: activation quantization
# I8: 8-bit signed integer 
# T: ternary
# S: symmetric 
# C: channelwise 
# D: dynamic scaling 
# STE: straight-through estimator


class QAI8SCD_STE(Function):
    @staticmethod
    def forward(ctx, x, outtype=None):
        if outtype is None:
            outtype = x.type()
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        qx = (x * scale).round().clamp_(-128, 127).type(outtype)
        ctx.save_for_backward(scale)
        return qx, scale.T, None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        scale, = ctx.saved_tensors
        # STE: Pass gradients straight-through, accounting for scaling
        return grad_output * scale , None

def QuantQAI8SCD_STE(x: torch.Tensor, outtype=None):
    return QAI8SCD_STE.apply(x, outtype)



class QAI8STD_STE(Function):
    @staticmethod
    def forward(ctx, x, outtype=None):
        if outtype is None:
            outtype = x.type()
        scale = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
        qx = (x * scale).round().clamp_(-128, 127).type(outtype)
        ctx.save_for_backward(scale)
        return qx, scale, None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        scale, = ctx.saved_tensors
        # STE: Pass gradients straight-through, accounting for scaling
        return grad_output * scale , None

def QuantQAI8STD_STE(x: torch.Tensor, outtype=None):
    return QAI8STD_STE.apply(x, outtype)


