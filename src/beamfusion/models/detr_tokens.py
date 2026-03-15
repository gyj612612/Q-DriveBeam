from __future__ import annotations

from pathlib import Path
from typing import Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTokenPooler(nn.Module):
    SUPPORTED = {"score_weighted_mean", "attn_pool", "cls_cross_attn"}

    def __init__(
        self,
        embed_dim: int,
        mode: str = "score_weighted_mean",
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.SUPPORTED:
            raise ValueError(f"Unsupported pool mode: {mode}. Choices: {sorted(self.SUPPORTED)}")
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode == "attn_pool":
            self.score_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
        elif mode == "cls_cross_attn":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        score_weights = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)

        if self.mode == "score_weighted_mean":
            pool_weights = score_weights
            scene = torch.sum(tokens * pool_weights.unsqueeze(-1), dim=1)
            return {"scene": scene, "pool_weights": pool_weights}

        if self.mode == "attn_pool":
            logits = self.score_mlp(tokens).squeeze(-1) + torch.log(scores.clamp_min(1e-6))
            pool_weights = F.softmax(logits, dim=1)
            scene = torch.sum(tokens * pool_weights.unsqueeze(-1), dim=1)
            return {"scene": scene, "pool_weights": pool_weights}

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        attn_out, attn_weights = self.cross_attn(cls, tokens, tokens)
        cls_feat = self.norm1(cls + self.dropout(attn_out))
        cls_feat = self.norm2(cls_feat + self.dropout(self.ffn(cls_feat)))
        scene = cls_feat.squeeze(1)
        return {
            "scene": scene,
            "pool_weights": score_weights,
            "attn_weights": attn_weights.squeeze(1),
        }


class DetrTokenEncoder(nn.Module):
    def __init__(
        self,
        detr_repo: str,
        variant: str = "detr_resnet50",
        pretrained: bool = True,
        topk_queries: int = 32,
        embed_dim: int = 256,
        freeze_detr: bool = True,
        pool_mode: str = "score_weighted_mean",
        pool_heads: int = 4,
        pool_dropout: float = 0.1,
        checkpoint_path: str | None = None,
        checkpoint_strict: bool = True,
    ) -> None:
        super().__init__()
        self.topk_queries = topk_queries
        self.freeze_detr = freeze_detr
        self.detr = self._load_model(
            detr_repo=detr_repo,
            variant=variant,
            pretrained=pretrained if not checkpoint_path else False,
        )
        self._load_checkpoint_if_given(
            model=self.detr,
            checkpoint_path=checkpoint_path,
            strict=checkpoint_strict,
        )
        self.token_proj = nn.LazyLinear(embed_dim)
        self.pooler = QueryTokenPooler(
            embed_dim=embed_dim,
            mode=pool_mode,
            num_heads=pool_heads,
            dropout=pool_dropout,
        )

        if freeze_detr:
            for p in self.detr.parameters():
                p.requires_grad = False
            self.detr.eval()

    def _load_model(self, detr_repo: str, variant: str, pretrained: bool) -> nn.Module:
        repo = Path(detr_repo)
        if repo.exists():
            return torch.hub.load(str(repo), variant, pretrained=pretrained, source="local")
        return torch.hub.load("facebookresearch/detr:main", variant, pretrained=pretrained)

    def _load_checkpoint_if_given(self, model: nn.Module, checkpoint_path: str | None, strict: bool) -> None:
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            warnings.warn(
                f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)} keys: {path}",
                stacklevel=2,
            )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.freeze_detr:
            self.detr.eval()

        with torch.set_grad_enabled(not self.freeze_detr):
            out = self.detr(images)

        logits = out["pred_logits"]  # [B, 100, C+1]
        boxes = out["pred_boxes"]  # [B, 100, 4]
        objectness = 1.0 - logits.softmax(dim=-1)[..., -1]
        top_scores, top_idx = torch.topk(objectness, k=min(self.topk_queries, objectness.size(1)), dim=1)

        bsz = logits.size(0)
        batch_ids = torch.arange(bsz, device=logits.device).unsqueeze(-1)
        top_logits = logits[batch_ids, top_idx]
        top_boxes = boxes[batch_ids, top_idx]

        token_input = torch.cat([top_logits, top_boxes], dim=-1)
        tokens = self.token_proj(token_input)
        pooled = self.pooler(tokens=tokens, scores=top_scores)

        out = {
            "tokens": tokens,
            "scene": pooled["scene"],
            "scores": top_scores,
            "pool_weights": pooled["pool_weights"],
        }
        if "attn_weights" in pooled:
            out["attn_weights"] = pooled["attn_weights"]
        return out
