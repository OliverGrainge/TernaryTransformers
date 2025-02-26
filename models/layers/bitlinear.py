import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Callable 


class BitLinear(nn.Linear): 
    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_quant_fn: Callable = None, activation_quant_fn: Callable = None):
        super().__init__(in_features, out_features, bias)
        self.weight_quant_fn = weight_quant_fn
        self.activation_quant_fn = activation_quant_fn
        self._deploy = False
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qw, ws, wz = self.weight_quant_fn(self.weight, outtype=None)
        qx, xs, xz = self.activation_quant_fn(input, outtype=None)
        qy = F.linear(qx, qw)
        print(qy.shape, xs.shape, ws.shape)
        dqy = qy / xs / ws
        return dqy



            
