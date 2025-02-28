import torch
import torch.nn as nn


class ImageClassificationHead(nn.Module):
    def __init__(self, num_classes: int, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(mlp_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        x = self.fc(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
