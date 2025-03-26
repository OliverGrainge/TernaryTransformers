import torch
import torch.nn as nn
from config import HeadConfig

class MLMHead(nn.Module):
    def __init__(self, head_config: HeadConfig):
        super().__init__()

        # Build MLP layers
        in_features = head_config.dim

        self.fc1 = nn.Linear(head_config.dim, head_config.dim)
        self.act = nn.GELU()
        self.ln1 = nn.LayerNorm(head_config.dim)
        self.fc2 = nn.Linear(head_config.dim, head_config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln1(x)
        x = self.fc2(x)
        return x
