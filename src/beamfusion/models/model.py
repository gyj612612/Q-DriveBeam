from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .detr_tokens import DetrTokenEncoder
from .encoders import FeatureDenoiseAutoEncoder, MLPEncoder
from .fusion import UncertaintyGatedFusion


class DetrIemfBeamModel(nn.Module):
    def __init__(
        self,
        gps_dim: int,
        power_dim: int,
        num_classes: int,
        embed_dim: int,
        dropout: float,
        detr_repo: str,
        detr_variant: str,
        detr_pretrained: bool,
        detr_checkpoint_path: str | None,
        detr_checkpoint_strict: bool,
        topk_queries: int,
        freeze_detr: bool,
        use_dual_view: bool,
        query_pool_mode: str = "score_weighted_mean",
        query_pool_heads: int = 4,
        modality_dropout_p: float = 0.0,
        ae_enabled: bool = False,
        ae_use_vae: bool = False,
        ae_latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.use_dual_view = use_dual_view
        self.modality_dropout_p = float(modality_dropout_p)
        self.ae_enabled = bool(ae_enabled)
        self.scene_encoder = DetrTokenEncoder(
            detr_repo=detr_repo,
            variant=detr_variant,
            pretrained=detr_pretrained,
            topk_queries=topk_queries,
            embed_dim=embed_dim,
            freeze_detr=freeze_detr,
            pool_mode=query_pool_mode,
            pool_heads=query_pool_heads,
            pool_dropout=dropout,
            checkpoint_path=detr_checkpoint_path,
            checkpoint_strict=detr_checkpoint_strict,
        )
        self.gps_encoder = MLPEncoder(gps_dim, embed_dim=embed_dim, dropout=dropout)
        self.power_encoder = MLPEncoder(power_dim, embed_dim=embed_dim, dropout=dropout)
        self.gps_ae = (
            FeatureDenoiseAutoEncoder(feature_dim=embed_dim, latent_dim=ae_latent_dim, use_vae=ae_use_vae)
            if self.ae_enabled
            else None
        )
        self.power_ae = (
            FeatureDenoiseAutoEncoder(feature_dim=embed_dim, latent_dim=ae_latent_dim, use_vae=ae_use_vae)
            if self.ae_enabled
            else None
        )
        self.fusion = UncertaintyGatedFusion(
            num_modalities=3,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.missing_tokens = nn.Parameter(torch.zeros(3, embed_dim))

    def _apply_modality_dropout(self, modalities: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        num_modalities = len(modalities)
        bsz = modalities[0].size(0)
        device = modalities[0].device

        if (not self.training) or self.modality_dropout_p <= 0:
            mask = torch.ones(bsz, num_modalities, device=device)
            return modalities, mask

        keep = (torch.rand(bsz, num_modalities, device=device) > self.modality_dropout_p)
        all_dropped = keep.sum(dim=1) == 0
        if all_dropped.any():
            fallback = torch.randint(0, num_modalities, (int(all_dropped.sum().item()),), device=device)
            keep[all_dropped] = False
            keep[all_dropped, fallback] = True

        dropped_modalities: list[torch.Tensor] = []
        for i, feat in enumerate(modalities):
            missing = self.missing_tokens[i].unsqueeze(0).expand_as(feat)
            dropped = torch.where(keep[:, i : i + 1], feat, missing)
            dropped_modalities.append(dropped)

        return dropped_modalities, keep.float()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "cached_scene" in batch:
            # Optional fast path: reuse precomputed scene features when DETR is frozen.
            scene = batch["cached_scene"]
            scene_main_out: Dict[str, torch.Tensor] = {}
        else:
            scene_main_out = self.scene_encoder(batch["image"])
            scene_main = scene_main_out["scene"]
            if self.use_dual_view:
                scene_aux = self.scene_encoder(batch["image_aux"])["scene"]
                scene = 0.5 * (scene_main + scene_aux)
            else:
                scene = scene_main

        gps = self.gps_encoder(batch["gps"])
        power = self.power_encoder(batch["power"])
        ae_rec_losses: list[torch.Tensor] = []
        ae_kl_losses: list[torch.Tensor] = []
        if self.ae_enabled:
            assert self.gps_ae is not None and self.power_ae is not None
            gps, gps_rec, gps_kl = self.gps_ae(gps)
            power, power_rec, power_kl = self.power_ae(power)
            ae_rec_losses.extend([gps_rec, power_rec])
            ae_kl_losses.extend([gps_kl, power_kl])

        modalities, modality_mask = self._apply_modality_dropout([scene, gps, power])
        out = self.fusion(modalities, modality_mask=modality_mask)
        out["scene_feat"] = modalities[0]
        out["gps_feat"] = modalities[1]
        out["power_feat"] = modalities[2]
        out["modality_mask"] = modality_mask
        if ae_rec_losses:
            out["ae_rec_loss"] = torch.stack(ae_rec_losses).mean()
            out["ae_kl_loss"] = torch.stack(ae_kl_losses).mean()
        else:
            out["ae_rec_loss"] = scene.new_tensor(0.0)
            out["ae_kl_loss"] = scene.new_tensor(0.0)
        if "pool_weights" in scene_main_out:
            out["scene_pool_weights"] = scene_main_out["pool_weights"]
        if "attn_weights" in scene_main_out:
            out["scene_attn_weights"] = scene_main_out["attn_weights"]
        return out
