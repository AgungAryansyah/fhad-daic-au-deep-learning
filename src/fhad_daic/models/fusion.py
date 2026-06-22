import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .tcn import TCNEncoder


class ReliabilityCalculator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        aux_v: dict[str, torch.Tensor],
        aux_a: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C_v = aux_v["confidence_mean"].unsqueeze(-1)
        C_v_norm = torch.clamp((C_v - 0.5) / 0.5, 0.0, 1.0)

        AU_dyn = aux_v["au_dyn_mean"].unsqueeze(-1)
        AU_dyn_norm = torch.tanh(AU_dyn * 10.0)

        pose_var = aux_v["pose_var_mean"].unsqueeze(-1)
        Pose_stab = 1.0 - torch.tanh(pose_var * 0.1)

        R_v = (C_v_norm + AU_dyn_norm + Pose_stab) / 3.0

        HNR_mean = aux_a["hnr_mean"].unsqueeze(-1)
        R_a = torch.tanh(HNR_mean / 10.0)

        weights = torch.softmax(torch.cat([R_v, R_a], dim=-1), dim=-1)
        w_v = weights[:, 0:1]
        w_a = weights[:, 1:2]

        return w_v, w_a


class FusionModel(nn.Module):
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
        self.reliability = ReliabilityCalculator()
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

        w_v, w_a = self.reliability(aux_v, aux_a)

        F = torch.cat([w_v * v, w_a * a], dim=-1)
        return self.classifier(F)


class FusionTCNModel(nn.Module):
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
        self.reliability = ReliabilityCalculator()
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

        w_v, w_a = self.reliability(aux_v, aux_a)

        F = torch.cat([w_v * V, w_a * A], dim=-1)
        return self.classifier(F)
