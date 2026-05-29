import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, num_inputs: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.classifier(out)
