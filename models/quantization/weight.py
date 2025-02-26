import torch 
from torch.autograd import Function

# I8: 8-bit signed integer 
# T: ternary
# S: symmetric 
# C: channelwise 
# D: dynamic scaling 
# STE: straight-through estimator

class QWI8SCD_STE(Function):
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

def QuantQWI8SCD_STE(x: torch.Tensor, outtype=None):
    return QWI8SCD_STE.apply(x, outtype)



class QWI8STD_STE(Function):
    @staticmethod
    def forward(ctx, x, outtype=None):
        if outtype is None:
            outtype = x.type()
        scale = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
        qx = (x * scale).round().clamp_(-128, 127).type(outtype)
        ctx.save_for_backward(scale)
        return qx, scale.T, None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        scale, = ctx.saved_tensors
        return grad_output * scale , None

def QuantQWI8STD_STE(x: torch.Tensor, outtype=None):
    return QWI8STD_STE.apply(x, outtype)



class QWTSTD_STE(Function):
    @staticmethod
    def forward(ctx, w, outtype=None):
        if outtype is None:
            outtype = w.type()
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        qw = (w * scale).round().clamp_(-1, 1).type(outtype)
        ctx.save_for_backward(scale)
        return qw, scale, None

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_none):
        scale, = ctx.saved_tensors
        return grad_output * scale, None

def QWTSTD_STE(w: torch.Tensor, outtype=None):
    return QWTSTD_STE.apply(w, outtype)



