import torch
import torch.nn as nn

from config import Config


class MLMHead(nn.Module):
    def __init__(self, model_config: Config):
        super().__init__()

        self.fc1 = nn.Linear(model_config.head_dim, model_config.head_dim)
        self.act = nn.GELU()
        self.ln1 = nn.LayerNorm(model_config.head_dim)
        self.fc2 = nn.Linear(model_config.head_dim, model_config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln1(x)
        x = self.fc2(x)
        return x
