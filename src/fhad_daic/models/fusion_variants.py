import torch
import torch.nn as nn

from .mlp import MLP
from .tcn import TCNEncoder


class ConcatFusionModel(nn.Module):
    def __init__(
        self,
        vis_dim: int,
        aud_dim: int,
        hidden_dims: list[int],
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.hidden_v = vis_dim // 2
        self.hidden_a = aud_dim // 2
        self.fusion_dim = self.hidden_v + self.hidden_a

        self.proj_v = nn.Linear(vis_dim, self.hidden_v)
        self.proj_a = nn.Linear(aud_dim, self.hidden_a)
        self.classifier = MLP(self.fusion_dim, hidden_dims, dropout, num_classes)

    def forward(
        self,
        F_v: torch.Tensor,
        F_a: torch.Tensor,
        aux_v: dict[str, torch.Tensor],
        aux_a: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        v = self.proj_v(F_v)
        a = self.proj_a(F_a)
        return self.classifier(torch.cat([v, a], dim=-1))


class ConcatFusionTCNModel(nn.Module):
    def __init__(
        self,
        vis_dim: int,
        aud_dim: int,
        vis_channels: list[int],
        aud_channels: list[int],
        kernel_size: int,
        tcn_dropout: float,
        fusion_hidden_dims: list[int],
        fusion_dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.vis_encoder = TCNEncoder(vis_dim, vis_channels, kernel_size, tcn_dropout)
        self.aud_encoder = TCNEncoder(aud_dim, aud_channels, kernel_size, tcn_dropout)

        vis_embed = vis_channels[-1]
        aud_embed = aud_channels[-1]
        proj_v_dim = vis_embed // 2
        proj_a_dim = aud_embed // 2
        self.fusion_dim = proj_v_dim + proj_a_dim

        self.proj_v = nn.Linear(vis_embed, proj_v_dim)
        self.proj_a = nn.Linear(aud_embed, proj_a_dim)
        self.classifier = MLP(self.fusion_dim, fusion_hidden_dims, fusion_dropout, num_classes)

    def forward(
        self,
        X_v: torch.Tensor,
        mask_v: torch.Tensor,
        X_a: torch.Tensor,
        mask_a: torch.Tensor,
        aux_v: dict[str, torch.Tensor],
        aux_a: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        V = self.proj_v(self.vis_encoder(X_v, mask_v))
        A = self.proj_a(self.aud_encoder(X_a, mask_a))
        return self.classifier(torch.cat([V, A], dim=-1))
