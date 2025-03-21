from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# More can be found about this layer here:https://arxiv.org/abs/2402.17764


class WeightQuantizer(Function):
    @staticmethod
    def forward(ctx, w, alpha, gamma):
        out_features, in_features = w.shape
        delta = (0.7 * w.abs().mean(dim=1, keepdim=True)).clamp(min=1e-5)
        scale = 1.0 / delta
        T = (w * scale).round().clamp_(-1, 1)
        D = alpha * T + gamma 
        ctx.save_for_backward(T, w, delta, alpha)
        return D
    
    @staticmethod
    def backward(ctx, grad_output):
        T, w, delta, alpha = ctx.saved_tensors
        zero_mask = (w.abs() <= delta)

        grad_w = grad_output * alpha * zero_mask
        grad_alpha = torch.sum(grad_output * T).reshape_as(alpha)
        grad_gamma = torch.sum(grad_output).reshape(1)
        return grad_w, grad_alpha, grad_gamma
    


def quant(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor):
    return WeightQuantizer.apply(x, alpha, gamma)


class TLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_scales()

    def _init_scales(self): 
        delta = (0.7 * self.weight.abs().mean())
        self.alpha.data.fill_(delta)
        self.gamma.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant(self.weight, self.alpha, self.gamma)
        output = F.linear(x, dqw, bias=self.bias)
        return output
    
