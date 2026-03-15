from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGatedFusion(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.prior_net = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_modalities),
        )
        self.uncertainty_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)) for _ in range(num_modalities)]
        )
        self.branch_heads = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_modalities)])
        self.fused_head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modalities: List[torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(modalities)}")

        concat = torch.cat(modalities, dim=-1)
        prior = F.softmax(self.prior_net(concat), dim=-1)

        uncert = []
        for head, feat in zip(self.uncertainty_heads, modalities):
            uncert.append(F.softplus(head(feat)) + 1e-4)
        uncert = torch.cat(uncert, dim=-1)
        inv_uncert = 1.0 / uncert

        weights = prior * inv_uncert
        if modality_mask is not None:
            if modality_mask.shape != weights.shape:
                raise ValueError(f"modality_mask shape {modality_mask.shape} != weight shape {weights.shape}")
            weights = weights * modality_mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        stacked = torch.stack(modalities, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        fused = self.dropout(fused)

        fused_logits = self.fused_head(fused)
        branch_logits = [head(feat) for head, feat in zip(self.branch_heads, modalities)]

        if modality_mask is None:
            target = weights.new_full((self.num_modalities,), 1.0 / self.num_modalities)
        else:
            active = modality_mask / (modality_mask.sum(dim=1, keepdim=True) + 1e-6)
            target = active.mean(dim=0)
        gate_reg = torch.mean((weights.mean(dim=0) - target) ** 2)

        return {
            "fused_logits": fused_logits,
            "branch_logits": branch_logits,
            "weights": weights,
            "gate_reg": gate_reg,
        }
