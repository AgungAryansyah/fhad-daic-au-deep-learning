import torch
import torch.nn as nn

from .tcn import TCNBlock


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
