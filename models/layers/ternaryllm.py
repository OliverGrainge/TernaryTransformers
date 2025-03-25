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
        zero_mask = w.abs() <= delta

        grad_w = grad_output * alpha * zero_mask
        grad_alpha = torch.sum(grad_output * T).reshape_as(alpha)
        grad_gamma = torch.sum(grad_output).reshape(1)
        return grad_w, grad_alpha, grad_gamma


def quant_tensor(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor):
    return WeightQuantizer.apply(x, alpha, gamma)


class TLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_scales()

    def _init_scales(self):
        delta = 0.7 * self.weight.abs().mean()
        self.alpha.data.fill_(delta)
        self.gamma.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant_tensor(self.weight, self.alpha, self.gamma)
        output = F.linear(x, dqw, bias=self.bias)
        return output









class WeightQuantizerChannel(Function):
    @staticmethod
    def forward(ctx, w, alpha, gamma):
        out_features, in_features = w.shape
        delta = (0.7 * w.abs().mean(dim=1, keepdim=True)).clamp(min=1e-5)
        scale = 1.0 / delta
        T = (w * scale).round().clamp_(-1, 1)
        D = alpha.view(-1, 1) * T + gamma.view(-1, 1)
        ctx.save_for_backward(T, w, delta, alpha)
        return D

    @staticmethod
    def backward(ctx, grad_output):
        T, w, delta, alpha = ctx.saved_tensors
        zero_mask = w.abs() <= delta

        grad_w = grad_output * alpha.view(-1, 1) * zero_mask
        grad_alpha = torch.sum(grad_output * T, dim=1).reshape_as(alpha)
        grad_gamma = torch.sum(grad_output, dim=1).reshape_as(alpha)
        return grad_w, grad_alpha, grad_gamma


def quant_channel(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor):
    return WeightQuantizerChannel.apply(x, alpha, gamma)


class TLinearChannel(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.zeros(out_features))
        self.gamma = nn.Parameter(torch.zeros(out_features))
        self._init_scales()

    def _init_scales(self):
        delta = 0.7 * self.weight.abs().mean(
            dim=1
        )  # This returns a tensor of shape [out_features]
        self.alpha.data = delta  # Use direct assignment instead of fill_
        self.gamma.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant_channel(self.weight, self.alpha, self.gamma)
        output = F.linear(x, dqw, bias=self.bias)
        return output







class WeightQuantizerGroup(Function):
    @staticmethod
    def forward(ctx, w, alpha, gamma, group_size):
        out_features, in_features = w.shape
        delta = (0.7 * w.abs().mean(dim=1, keepdim=True)).clamp(min=1e-5)
        scale = 1.0 / delta
        T = (w * scale).round().clamp_(-1, 1)
        D = alpha.view(-1, 1) * T.view(-1, group_size) + gamma.view(-1, 1)
        ctx.save_for_backward(T, w, delta, alpha, group_size)
        return D.view(out_features, in_features)

    @staticmethod
    def backward(ctx, grad_output):
        T, w, delta, alpha, group_size = ctx.saved_tensors
        zero_mask = w.abs() <= delta

        grad_w = (
            grad_output.view(-1, group_size)
            * alpha.view(-1, 1)
            * zero_mask.view(-1, group_size)
        ).view_as(grad_output)
        grad_alpha = torch.sum(
            (grad_output * T).view(-1, group_size), dim=1
        ).reshape_as(alpha)
        grad_gamma = torch.sum(grad_output.view(-1, group_size), dim=1).reshape_as(
            alpha
        )
        return grad_w, grad_alpha, grad_gamma, None


def quant_group(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, group_size: int):
    return WeightQuantizerGroup.apply(x, alpha, gamma, group_size)


class TLinearGroup(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, group_size: int = 64
    ):
        super().__init__(in_features, out_features, bias)
        assert (
            out_features % group_size == 0
        ), "out_features must be divisible by group_size"
    
        self.alpha = nn.Parameter(
            torch.zeros(out_features * (in_features // group_size))
        )
        self.gamma = nn.Parameter(
            torch.zeros(out_features * (in_features // group_size))
        )
        self.group_size = torch.tensor([group_size])
        self._init_scales()


    def _init_scales(self):
        delta = (
            (0.7 * self.weight.abs().mean(dim=1))
            .view(-1, 1)
            .repeat(1, self.in_features // self.group_size)
            .flatten()
        )  # This returns a tensor of shape [out_features]
        self.alpha.data = delta  # Use direct assignment instead of fill_
        self.gamma.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqw = quant_group(self.weight, self.alpha, self.gamma, self.group_size)
        output = F.linear(x, dqw, bias=self.bias)
        return output