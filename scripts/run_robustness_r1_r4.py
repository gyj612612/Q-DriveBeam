from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beamfusion.config import TrainConfig
from beamfusion.data import Scenario36Dataset, prepare_scenario36
from beamfusion.models import DetrIemfBeamModel
from beamfusion.train import _autocast_dtype, _resolve_num_workers, evaluate
from beamfusion.utils import ensure_dir, set_seed, write_json


class StressDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        mode: str,
        seed: int,
        blur_sigma: float = 0.0,
        occlusion: float = 0.0,
        gps_noise_std_norm: float = 0.0,
        power_mask_ratio: float = 0.0,
        missing_mode: Optional[str] = None,
    ) -> None:
        self.base = base
        self.mode = mode
        self.seed = int(seed)
        self.blur_sigma = float(blur_sigma)
        self.occlusion = float(occlusion)
        self.gps_noise_std_norm = float(gps_noise_std_norm)
        self.power_mask_ratio = float(power_mask_ratio)
        self.missing_mode = missing_mode

    def __len__(self) -> int:
        return len(self.base)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx * 7919)

    def _apply_camera(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        out = img.clone()
        if self.blur_sigma > 0:
            k = max(3, int(round(self.blur_sigma * 3) * 2 + 1))
            out = TF.gaussian_blur(out, [k, k], [self.blur_sigma, self.blur_sigma])
        if self.occlusion > 0:
            h, w = out.shape[-2], out.shape[-1]
            area = max(1, int(h * w * self.occlusion))
            side = int(np.sqrt(area))
            side = max(1, min(side, h, w))
            y0 = int(rng.integers(0, max(1, h - side + 1)))
            x0 = int(rng.integers(0, max(1, w - side + 1)))
            out[:, y0 : y0 + side, x0 : x0 + side] = 0.0
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.base[idx]
        out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
        rng = self._rng(idx)

        if self.mode == "r1":
            out["image"] = self._apply_camera(out["image"], rng)
            out["image_aux"] = self._apply_camera(out["image_aux"], rng)
        elif self.mode == "r2":
            if self.gps_noise_std_norm > 0:
                n = torch.randn_like(out["gps"]) * self.gps_noise_std_norm
                out["gps"] = out["gps"] + n
        elif self.mode == "r3":
            if self.power_mask_ratio > 0:
                d = out["power"].numel()
                k = max(1, int(d * self.power_mask_ratio))
                idxs = rng.choice(d, size=k, replace=False)
                p = out["power"].view(-1)
                p[idxs] = 0.0
        elif self.mode == "r4":
            m = self.missing_mode or "mixed"
            if m == "mixed":
                m = ["camera", "gps", "power"][int(rng.integers(0, 3))]
            if m == "camera":
                out["image"].zero_()
                out["image_aux"].zero_()
            elif m == "gps":
                out["gps"].zero_()
            elif m == "power":
                out["power"].zero_()

        return out


def _load_cfg(cfg_path: Path) -> TrainConfig:
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def _build_model(cfg: TrainConfig, prepared) -> DetrIemfBeamModel:
    num_classes = int(prepared.labels.max() + 1)
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
        modality_dropout_p=0.0,
        ae_enabled=cfg.ae_enabled,
        ae_use_vae=cfg.ae_use_vae,
        ae_latent_dim=cfg.ae_latent_dim,
    )
    return model


def _pick_best_run(a7_results_json: Path) -> Dict[str, Any]:
    data = json.loads(a7_results_json.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows in {a7_results_json}")
    best = sorted(rows, key=lambda r: float(r["summary"]["test_acc"]), reverse=True)[0]
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1-R4 robustness evaluation from best A7 checkpoint.")
    parser.add_argument("--a7-results-json", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs" / "robustness_r1_r4"))
    parser.add_argument("--tag", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    best = _pick_best_run(Path(args.a7_results_json))
    run_dir = Path(best["output_dir"])
    cfg = _load_cfg(run_dir / "config.json")

    out_dir = Path(args.output_root) / args.tag
    ensure_dir(out_dir)
    write_json(out_dir / "selected_best_run.json", best)

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
    base_test = Scenario36Dataset(prepared, "test", image_size=cfg.image_size, use_dual_view=cfg.use_dual_view)

    device = torch.device(cfg.device)
    model = _build_model(cfg, prepared).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    num_workers = _resolve_num_workers(cfg)
    common = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": cfg.device.startswith("cuda"),
        "drop_last": False,
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(cfg.persistent_workers)
        common["prefetch_factor"] = int(cfg.prefetch_factor)

    xy_std = float(np.mean([prepared.gps_std[0], prepared.gps_std[1], prepared.gps_std[6], prepared.gps_std[7]]))
    rows: List[Dict[str, Any]] = []

    def run_case(case_id: str, ds: Dataset, meta: Dict[str, Any]) -> None:
        loader = DataLoader(ds, **common)
        m = evaluate(
            model,
            loader,
            device=device,
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
            max_steps=args.max_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        row = {"case_id": case_id, **meta, **m}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    run_case("baseline", base_test, {"group": "baseline"})

    for sigma in [1.0, 2.0, 3.0]:
        for occ in [0.1, 0.2, 0.3]:
            ds = StressDataset(base_test, mode="r1", seed=args.seed, blur_sigma=sigma, occlusion=occ)
            run_case(f"r1_blur{sigma}_occ{occ}", ds, {"group": "R1", "blur_sigma": sigma, "occlusion": occ})

    for std_m in [0.5, 1.0, 2.0, 5.0]:
        std_norm = float(std_m / max(1e-6, xy_std))
        ds = StressDataset(base_test, mode="r2", seed=args.seed, gps_noise_std_norm=std_norm)
        run_case(f"r2_gps{std_m}m", ds, {"group": "R2", "gps_noise_m": std_m, "gps_noise_norm": std_norm})

    for ratio in [0.1, 0.2, 0.3]:
        ds = StressDataset(base_test, mode="r3", seed=args.seed, power_mask_ratio=ratio)
        run_case(f"r3_powermask{ratio}", ds, {"group": "R3", "power_mask_ratio": ratio})

    for m in ["camera", "gps", "power", "mixed"]:
        ds = StressDataset(base_test, mode="r4", seed=args.seed, missing_mode=m)
        run_case(f"r4_missing_{m}", ds, {"group": "R4", "missing_mode": m})

    write_json(out_dir / "results.json", {"rows": rows, "a7_results_json": args.a7_results_json, "selected_best_run": best})

    md = [
        f"# Robustness R1-R4 ({args.tag})",
        "",
        "| case_id | group | acc | loss |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['case_id']} | {r.get('group','')} | {float(r['acc']):.4f} | {float(r['loss']):.4f} |")
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "docs" / "robustness_r1_r4_latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
