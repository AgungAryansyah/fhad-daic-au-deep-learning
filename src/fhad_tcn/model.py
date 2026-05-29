import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, : -self.padding] if self.padding > 0 else self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int, dropout: float, num_classes: int):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.mean(dim=-1)
        return self.classifier(x)


class MILTCN(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int, dropout: float, num_classes: int, attn_dim: int = 64):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.feature_extractor = nn.Sequential(*layers)
        self.attn_V = nn.Linear(num_channels[-1], attn_dim)
        self.attn_U = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.mean(dim=-1)

        if len(session_ids) == 0:
            return x.new_zeros(0, self.classifier.out_features)

        unique_sids, inverse = torch.unique(session_ids, sorted=True, return_inverse=True)
        n_bags = len(unique_sids)

        bag_embeddings = x.new_zeros(n_bags, x.size(1))
        for b in range(n_bags):
            mask = inverse == b
            bag_h = x[mask]
            if bag_h.size(0) == 1:
                bag_embeddings[b] = bag_h[0]
            else:
                attn_h = torch.tanh(self.attn_V(bag_h))
                attn = torch.softmax(self.attn_U(attn_h), dim=0)
                bag_embeddings[b] = (attn * bag_h).sum(dim=0)

        return self.classifier(bag_embeddings)


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
