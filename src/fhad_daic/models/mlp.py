import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_features: int, hidden_dims: list[int], dropout: float, num_classes: int):
        super().__init__()
        layers = []
        prev = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
