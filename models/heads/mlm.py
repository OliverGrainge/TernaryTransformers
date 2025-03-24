import torch
import torch.nn as nn


class MLMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()

        # Build MLP layers
        in_features = dim

        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln1(x)
        x = self.fc2(x)
        return x
