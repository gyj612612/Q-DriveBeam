from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig
from .data import Scenario36Dataset, prepare_scenario36
from .losses import compute_losses
from .models import DetrIemfBeamModel
from .utils import append_jsonl, ensure_dir, set_seed, write_json


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = int(max(1, min(k, logits.size(1))))
    topk = logits.topk(k=k, dim=1).indices
    hit = topk.eq(labels.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean().item())


def _autocast_dtype(name: str) -> torch.dtype:
    n = str(name).lower().strip()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _resolve_num_workers(cfg: TrainConfig) -> int:
    if cfg.num_workers is not None:
        return int(cfg.num_workers)
    cpu = os.cpu_count() or 8
    return max(2, min(8, cpu // 2))


def _setup_runtime(cfg: TrainConfig) -> None:
    if cfg.cpu_threads > 0:
        torch.set_num_threads(int(cfg.cpu_threads))

    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)


class CachedSceneDataset(Dataset):
    def __init__(
        self,
        scene: torch.Tensor,
        gps: torch.Tensor,
        power: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.scene = scene
        self.gps = gps
        self.power = power
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "cached_scene": self.scene[idx],
            "gps": self.gps[idx],
            "power": self.power[idx],
            "label": self.labels[idx],
        }


@torch.no_grad()
def _cache_scene_features(
    model: DetrIemfBeamModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_dual_view: bool,
) -> CachedSceneDataset:
    loader_kwargs = {
        "batch_size": int(max(1, batch_size)),
        "shuffle": False,
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(max(1, prefetch_factor))

    loader = DataLoader(dataset, **loader_kwargs)
    model.eval()

    scene_feats = []
    gps_feats = []
    power_feats = []
    labels = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            main = model.scene_encoder(batch["image"])["scene"]
            if use_dual_view:
                aux = model.scene_encoder(batch["image_aux"])["scene"]
                scene = 0.5 * (main + aux)
            else:
                scene = main

        scene_feats.append(scene.detach().cpu())
        gps_feats.append(batch["gps"].detach().cpu())
        power_feats.append(batch["power"].detach().cpu())
        labels.append(batch["label"].detach().cpu())

    return CachedSceneDataset(
        scene=torch.cat(scene_feats, dim=0).contiguous(),
        gps=torch.cat(gps_feats, dim=0).contiguous(),
        power=torch.cat(power_feats, dim=0).contiguous(),
        labels=torch.cat(labels, dim=0).contiguous(),
    )


@torch.no_grad()
def evaluate(
    model: DetrIemfBeamModel,
    loader: DataLoader,
    device: torch.device,
    branch_aux_lambda: float,
    consistency_lambda: float,
    gate_reg_lambda: float,
    iemf_enabled: bool = False,
    iemf_psai: float = 1.2,
    iemf_scale_min: float = 0.5,
    iemf_scale_max: float = 2.0,
    iemf_detach_coeff: bool = True,
    ae_recon_lambda: float = 0.0,
    ae_kl_lambda: float = 0.0,
    max_steps: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    model.eval()
    total_sum = 0.0
    main_sum = 0.0
    branch_sum = 0.0
    consistency_sum = 0.0
    gate_reg_sum = 0.0
    ae_rec_sum = 0.0
    ae_kl_sum = 0.0
    iemf_coeff_sum = 0.0
    acc1_sum = 0.0
    acc3_sum = 0.0
    acc5_sum = 0.0
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            out = model(batch)
            losses = compute_losses(
                out,
                batch["label"],
                branch_aux_lambda=branch_aux_lambda,
                consistency_lambda=consistency_lambda,
                gate_reg_lambda=gate_reg_lambda,
                iemf_enabled=iemf_enabled,
                iemf_psai=iemf_psai,
                iemf_scale_min=iemf_scale_min,
                iemf_scale_max=iemf_scale_max,
                iemf_detach_coeff=iemf_detach_coeff,
                ae_recon_lambda=ae_recon_lambda,
                ae_kl_lambda=ae_kl_lambda,
            )
        bs = batch["label"].size(0)
        total_sum += float(losses["total"].item()) * bs
        main_sum += float(losses["main"].item()) * bs
        branch_sum += float(losses["branch"].item()) * bs
        consistency_sum += float(losses["consistency"].item()) * bs
        gate_reg_sum += float(losses["gate_reg"].item()) * bs
        ae_rec_sum += float(losses["ae_rec"].item()) * bs
        ae_kl_sum += float(losses["ae_kl"].item()) * bs
        iemf_coeff_sum += float(losses["iemf_coeff_mean"].item()) * bs
        acc1_sum += _topk_accuracy(out["fused_logits"], batch["label"], k=1) * bs
        acc3_sum += _topk_accuracy(out["fused_logits"], batch["label"], k=3) * bs
        acc5_sum += _topk_accuracy(out["fused_logits"], batch["label"], k=5) * bs
        n += bs

    d = max(1, n)
    return {
        "loss": total_sum / d,
        "acc": acc1_sum / d,
        "top1": acc1_sum / d,
        "top3": acc3_sum / d,
        "top5": acc5_sum / d,
        "main": main_sum / d,
        "branch": branch_sum / d,
        "consistency": consistency_sum / d,
        "gate_reg": gate_reg_sum / d,
        "ae_rec": ae_rec_sum / d,
        "ae_kl": ae_kl_sum / d,
        "iemf_coeff": iemf_coeff_sum / d,
    }


def train(cfg: TrainConfig) -> Dict[str, float]:
    start_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _setup_runtime(cfg)
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(
        out_dir / "run_meta.json",
        {
            "start_time_utc": start_time,
            "seed": cfg.seed,
            "device": cfg.device,
        },
    )

    prepared = prepare_scenario36(
        scenario_root=cfg.scenario_root,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        image_key_a=cfg.image_key_a,
        image_key_b=cfg.image_key_b,
        power_use_log=cfg.power_use_log,
        power_log_clip_min=cfg.power_log_clip_min,
        max_samples=cfg.max_samples,
    )
    num_classes = int(prepared.labels.max() + 1)

    train_ds = Scenario36Dataset(prepared, "train", image_size=cfg.image_size, use_dual_view=cfg.use_dual_view)
    val_ds = Scenario36Dataset(prepared, "val", image_size=cfg.image_size, use_dual_view=cfg.use_dual_view)
    test_ds = Scenario36Dataset(prepared, "test", image_size=cfg.image_size, use_dual_view=cfg.use_dual_view)

    device = torch.device(cfg.device)
    model = DetrIemfBeamModel(
        gps_dim=prepared.gps.shape[1],
        power_dim=prepared.power.shape[1],
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        detr_repo=cfg.detr_repo,
        detr_variant=cfg.detr_variant,
        detr_pretrained=cfg.detr_pretrained,
        detr_checkpoint_path=cfg.detr_checkpoint_path,
        detr_checkpoint_strict=cfg.detr_checkpoint_strict,
        topk_queries=cfg.topk_queries,
        freeze_detr=cfg.freeze_detr,
        use_dual_view=cfg.use_dual_view,
        query_pool_mode=cfg.query_pool_mode,
        query_pool_heads=cfg.query_pool_heads,
        modality_dropout_p=cfg.modality_dropout_p,
        ae_enabled=cfg.ae_enabled,
        ae_use_vae=cfg.ae_use_vae,
        ae_latent_dim=cfg.ae_latent_dim,
    ).to(device)

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    num_workers = _resolve_num_workers(cfg)
    pin_memory = cfg.device.startswith("cuda")

    cache_scene = bool(cfg.cache_scene_features and cfg.freeze_detr)
    cache_build_seconds = 0.0
    if cache_scene:
        t0 = time.perf_counter()
        cache_workers = max(0, min(num_workers, 4))
        train_ds = _cache_scene_features(
            model=model,
            dataset=train_ds,
            device=device,
            batch_size=int(max(cfg.batch_size, cfg.cache_batch_size)),
            num_workers=cache_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_dual_view=cfg.use_dual_view,
        )
        val_ds = _cache_scene_features(
            model=model,
            dataset=val_ds,
            device=device,
            batch_size=int(max(cfg.batch_size, cfg.cache_batch_size)),
            num_workers=cache_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_dual_view=cfg.use_dual_view,
        )
        test_ds = _cache_scene_features(
            model=model,
            dataset=test_ds,
            device=device,
            batch_size=int(max(cfg.batch_size, cfg.cache_batch_size)),
            num_workers=cache_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_dual_view=cfg.use_dual_view,
        )
        cache_build_seconds = float(time.perf_counter() - t0)

    loader_workers = 0 if cache_scene else num_workers
    common_loader = {
        "batch_size": cfg.batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if loader_workers > 0:
        common_loader["persistent_workers"] = bool(cfg.persistent_workers)
        common_loader["prefetch_factor"] = int(cfg.prefetch_factor)

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    best_val = -1.0
    best_epoch = 0
    best_path = out_dir / "checkpoints" / "best.pt"
    log_path = out_dir / "train_log.jsonl"
    no_improve = 0
    stopped_by_patience = False
    stopped_by_wall_time = False
    epochs_completed = 0
    train_t0 = time.perf_counter()

    def wall_time_exceeded() -> bool:
        if cfg.max_wall_time_min is None:
            return False
        elapsed_min = (time.perf_counter() - train_t0) / 60.0
        return elapsed_min >= float(cfg.max_wall_time_min)

    for epoch in range(1, cfg.epochs + 1):
        if wall_time_exceeded():
            stopped_by_wall_time = True
            break
        model.train()
        epoch_total = []
        epoch_main = []
        epoch_branch = []
        epoch_consistency = []
        epoch_gate_reg = []
        epoch_ae_rec = []
        epoch_ae_kl = []
        epoch_iemf_coeff = []
        epoch_acc = []
        epoch_top3 = []
        epoch_top5 = []

        for step, batch in enumerate(train_loader):
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break
            if wall_time_exceeded():
                stopped_by_wall_time = True
                break
            batch = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(batch)
                losses = compute_losses(
                    out,
                    batch["label"],
                    branch_aux_lambda=cfg.branch_aux_lambda,
                    consistency_lambda=cfg.consistency_lambda,
                    gate_reg_lambda=cfg.gate_reg_lambda,
                    iemf_enabled=cfg.iemf_enabled,
                    iemf_psai=cfg.iemf_psai,
                    iemf_scale_min=cfg.iemf_scale_min,
                    iemf_scale_max=cfg.iemf_scale_max,
                    iemf_detach_coeff=cfg.iemf_detach_coeff,
                    ae_recon_lambda=cfg.ae_recon_lambda,
                    ae_kl_lambda=cfg.ae_kl_lambda,
                )
            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(losses["total"]).backward()
            else:
                losses["total"].backward()
            if cfg.grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if use_amp:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            epoch_total.append(float(losses["total"].item()))
            epoch_main.append(float(losses["main"].item()))
            epoch_branch.append(float(losses["branch"].item()))
            epoch_consistency.append(float(losses["consistency"].item()))
            epoch_gate_reg.append(float(losses["gate_reg"].item()))
            epoch_ae_rec.append(float(losses["ae_rec"].item()))
            epoch_ae_kl.append(float(losses["ae_kl"].item()))
            epoch_iemf_coeff.append(float(losses["iemf_coeff_mean"].item()))
            epoch_acc.append(_accuracy(out["fused_logits"], batch["label"]))
            epoch_top3.append(_topk_accuracy(out["fused_logits"], batch["label"], k=3))
            epoch_top5.append(_topk_accuracy(out["fused_logits"], batch["label"], k=5))

        epochs_completed = epoch
        train_metrics = {
            "loss": float(np.mean(epoch_total) if epoch_total else 0.0),
            "acc": float(np.mean(epoch_acc) if epoch_acc else 0.0),
            "top1": float(np.mean(epoch_acc) if epoch_acc else 0.0),
            "top3": float(np.mean(epoch_top3) if epoch_top3 else 0.0),
            "top5": float(np.mean(epoch_top5) if epoch_top5 else 0.0),
            "main": float(np.mean(epoch_main) if epoch_main else 0.0),
            "branch": float(np.mean(epoch_branch) if epoch_branch else 0.0),
            "consistency": float(np.mean(epoch_consistency) if epoch_consistency else 0.0),
            "gate_reg": float(np.mean(epoch_gate_reg) if epoch_gate_reg else 0.0),
            "ae_rec": float(np.mean(epoch_ae_rec) if epoch_ae_rec else 0.0),
            "ae_kl": float(np.mean(epoch_ae_kl) if epoch_ae_kl else 0.0),
            "iemf_coeff": float(np.mean(epoch_iemf_coeff) if epoch_iemf_coeff else 1.0),
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            branch_aux_lambda=cfg.branch_aux_lambda,
            consistency_lambda=cfg.consistency_lambda,
            gate_reg_lambda=cfg.gate_reg_lambda,
            iemf_enabled=cfg.iemf_enabled,
            iemf_psai=cfg.iemf_psai,
            iemf_scale_min=cfg.iemf_scale_min,
            iemf_scale_max=cfg.iemf_scale_max,
            iemf_detach_coeff=cfg.iemf_detach_coeff,
            ae_recon_lambda=cfg.ae_recon_lambda,
            ae_kl_lambda=cfg.ae_kl_lambda,
            max_steps=cfg.max_val_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "wall_time_min": float((time.perf_counter() - train_t0) / 60.0),
        }
        append_jsonl(log_path, row)

        if val_metrics["acc"] > (best_val + float(cfg.early_stop_min_delta)):
            best_val = val_metrics["acc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val}, best_path)
        else:
            no_improve += 1

        if cfg.early_stop_patience > 0 and no_improve >= int(cfg.early_stop_patience):
            stopped_by_patience = True
            break

    if not best_path.exists():
        best_epoch = max(1, epochs_completed)
        torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_acc": best_val}, best_path)

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        branch_aux_lambda=cfg.branch_aux_lambda,
        consistency_lambda=cfg.consistency_lambda,
        gate_reg_lambda=cfg.gate_reg_lambda,
        iemf_enabled=cfg.iemf_enabled,
        iemf_psai=cfg.iemf_psai,
        iemf_scale_min=cfg.iemf_scale_min,
        iemf_scale_max=cfg.iemf_scale_max,
        iemf_detach_coeff=cfg.iemf_detach_coeff,
        ae_recon_lambda=cfg.ae_recon_lambda,
        ae_kl_lambda=cfg.ae_kl_lambda,
        max_steps=cfg.max_val_steps,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    summary = {
        "best_val_acc": float(best_val),
        "best_epoch": int(best_ckpt["epoch"]),
        "test_acc": float(test_metrics["acc"]),
        "test_top1": float(test_metrics["top1"]),
        "test_top3": float(test_metrics["top3"]),
        "test_top5": float(test_metrics["top5"]),
        "test_loss": float(test_metrics["loss"]),
        "test_main": float(test_metrics["main"]),
        "test_branch": float(test_metrics["branch"]),
        "test_consistency": float(test_metrics["consistency"]),
        "test_gate_reg": float(test_metrics["gate_reg"]),
        "test_ae_rec": float(test_metrics["ae_rec"]),
        "test_ae_kl": float(test_metrics["ae_kl"]),
        "test_iemf_coeff": float(test_metrics["iemf_coeff"]),
        "num_classes": int(num_classes),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "start_time_utc": start_time,
        "end_time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device_used": str(device),
        "num_workers_used": int(loader_workers),
        "use_amp": bool(use_amp),
        "cache_scene_features_used": bool(cache_scene),
        "cache_build_seconds": float(cache_build_seconds),
        "effective_epochs": int(epochs_completed),
        "stopped_by_early_stop": bool(stopped_by_patience),
        "stopped_by_wall_time": bool(stopped_by_wall_time),
        "wall_time_min": float((time.perf_counter() - train_t0) / 60.0),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
