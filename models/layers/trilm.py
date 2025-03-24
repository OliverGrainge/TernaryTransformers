from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# More can be found about this layer here:https://arxiv.org/abs/2402.17764


class WeightQuantizer(Function):
    @staticmethod
    def forward(ctx, w):
        delta = 1e-5 + w.abs().mean()
        scale = 1.0 / delta
        qw = (w * scale).round().clamp_(-1, 1)
        dqw = delta * qw
        ctx.save_for_backward(delta)
        return dqw

    @staticmethod
    def backward(ctx, grad_output):
        (delta,) = ctx.saved_tensors
        return grad_output.clone()


def quant(x: torch.Tensor):
    return WeightQuantizer.apply(x)


class TriLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant(self.weight)
        output = F.linear(x, dqw, bias=self.bias)
        return output
