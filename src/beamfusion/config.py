from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class TrainConfig:
    seed: int = 2026
    scenario_root: str = r"E:/6G/scenario36_merged"
    output_dir: str = r"E:/6G/Code/outputs/default"

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    batch_size: int = 8
    num_workers: Optional[int] = None
    persistent_workers: bool = True
    prefetch_factor: int = 2
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    max_samples: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None

    image_size: int = 224
    use_dual_view: bool = True
    image_key_a: str = "unit1_rgb5"
    image_key_b: str = "unit1_rgb6"
    power_use_log: bool = False
    power_log_clip_min: float = 1e-6

    detr_repo: str = r"E:/6G/detr-main"
    detr_variant: str = "detr_resnet50"
    detr_pretrained: bool = True
    detr_checkpoint_path: Optional[str] = None
    detr_checkpoint_strict: bool = True
    freeze_detr: bool = True
    topk_queries: int = 32
    query_pool_mode: str = "score_weighted_mean"
    query_pool_heads: int = 4
    cache_scene_features: bool = False
    cache_batch_size: int = 48

    embed_dim: int = 256
    dropout: float = 0.1
    modality_dropout_p: float = 0.0
    consistency_lambda: float = 0.1
    branch_aux_lambda: float = 0.4
    gate_reg_lambda: float = 1e-3
    iemf_enabled: bool = False
    iemf_psai: float = 1.2
    iemf_scale_min: float = 0.5
    iemf_scale_max: float = 2.0
    iemf_detach_coeff: bool = True
    ae_enabled: bool = False
    ae_use_vae: bool = False
    ae_latent_dim: int = 128
    ae_recon_lambda: float = 0.0
    ae_kl_lambda: float = 0.0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    max_wall_time_min: Optional[float] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    amp_dtype: str = "float16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cpu_threads: int = max(1, (os.cpu_count() or 8) - 2)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def _merge_dict(cfg: TrainConfig, update: Dict[str, Any]) -> TrainConfig:
    data = cfg.as_dict()
    for key, value in update.items():
        if key in data:
            data[key] = value
    return TrainConfig(**data)


def load_config(config_path: Optional[str] = None, override: Optional[Dict[str, Any]] = None) -> TrainConfig:
    cfg = TrainConfig()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        cfg = _merge_dict(cfg, parsed)
    if override:
        cfg = _merge_dict(cfg, override)
    return cfg
