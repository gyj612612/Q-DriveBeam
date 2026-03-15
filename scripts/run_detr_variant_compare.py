from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beamfusion.config import load_config
from beamfusion.train import train
from beamfusion.utils import append_jsonl, ensure_dir, write_json


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("No valid seeds parsed from --seeds.")
    return out


def _parse_str_list(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Parsed empty list.")
    return out


def _run_one(config_path: Path, output_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config(str(config_path), override={**overrides, "output_dir": str(output_dir)})
    return train(cfg)


def _aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        v = r["variant_id"]
        groups.setdefault(v, []).append(r)

    out: List[Dict[str, Any]] = []
    for vid, rs in sorted(groups.items()):
        val = [float(r["summary"]["best_val_acc"]) for r in rs]
        top1 = [float(r["summary"].get("test_top1", r["summary"]["test_acc"])) for r in rs]
        top3 = [float(r["summary"].get("test_top3", r["summary"]["test_acc"])) for r in rs]
        top5 = [float(r["summary"].get("test_top5", r["summary"]["test_acc"])) for r in rs]
        loss = [float(r["summary"]["test_loss"]) for r in rs]
        base = rs[0]
        out.append(
            {
                "variant_id": vid,
                "variant_name": base["variant_name"],
                "detr_variant": base["params"]["detr_variant"],
                "detr_checkpoint_path": base["params"].get("detr_checkpoint_path"),
                "num_seeds": len(rs),
                "seeds": [int(r["params"]["seed"]) for r in rs],
                "best_val_acc_mean": _mean(val),
                "best_val_acc_std": _std(val),
                "test_top1_mean": _mean(top1),
                "test_top1_std": _std(top1),
                "test_top3_mean": _mean(top3),
                "test_top3_std": _std(top3),
                "test_top5_mean": _mean(top5),
                "test_top5_std": _std(top5),
                "test_loss_mean": _mean(loss),
                "test_loss_std": _std(loss),
            }
        )
    return out


def _md_agg(rows: Sequence[Dict[str, Any]]) -> str:
    head = "| Variant | detr_variant | checkpoint | num_seeds | best_val_acc (mean+/-std) | test_top1 (mean+/-std) | test_top3 (mean+/-std) | test_top5 (mean+/-std) | test_loss (mean+/-std) |\n"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for r in rows:
        ckpt = r["detr_checkpoint_path"] if r["detr_checkpoint_path"] else "None"
        body.append(
            "| {name} | {dv} | `{ckpt}` | {n} | {vmean:.4f} +/- {vstd:.4f} | {t1m:.4f} +/- {t1s:.4f} | {t3m:.4f} +/- {t3s:.4f} | {t5m:.4f} +/- {t5s:.4f} | {lmean:.4f} +/- {lstd:.4f} |".format(
                name=r["variant_name"],
                dv=r["detr_variant"],
                ckpt=ckpt,
                n=int(r["num_seeds"]),
                vmean=float(r["best_val_acc_mean"]),
                vstd=float(r["best_val_acc_std"]),
                t1m=float(r["test_top1_mean"]),
                t1s=float(r["test_top1_std"]),
                t3m=float(r["test_top3_mean"]),
                t3s=float(r["test_top3_std"]),
                t5m=float(r["test_top5_mean"]),
                t5s=float(r["test_top5_std"]),
                lmean=float(r["test_loss_mean"]),
                lstd=float(r["test_loss_std"]),
            )
        )
    return head + sep + "\n".join(body) + "\n"


def _md_raw(rows: Sequence[Dict[str, Any]]) -> str:
    head = "| Run | Variant | Seed | detr_variant | test_top1 | test_top3 | test_top5 | test_loss | output_dir |\n"
    sep = "|---|---|---:|---|---:|---:|---:|---:|---|\n"
    body = []
    for r in rows:
        body.append(
            "| {rid} | {vname} | {seed} | {dv} | {top1:.4f} | {top3:.4f} | {top5:.4f} | {loss:.4f} | `{out}` |".format(
                rid=r["run_id"],
                vname=r["variant_name"],
                seed=int(r["params"]["seed"]),
                dv=r["params"]["detr_variant"],
                top1=float(r["summary"].get("test_top1", r["summary"]["test_acc"])),
                top3=float(r["summary"].get("test_top3", r["summary"]["test_acc"])),
                top5=float(r["summary"].get("test_top5", r["summary"]["test_acc"])),
                loss=float(r["summary"]["test_loss"]),
                out=r["output_dir"],
            )
        )
    return head + sep + "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DETR backbone/checkpoint variants with multi-seed aggregation.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "scenario36_detr_fusion.yaml"))
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs" / "detr_variant_compare"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seeds", type=str, default="2026")
    parser.add_argument("--variants", type=str, default="v_r50,v_r101,v_r101_dc5")
    parser.add_argument("--budget-mode", type=str, default="none", choices=["none", "fast_1h"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-panoptic", action="store_true")
    parser.add_argument("--resume-skip", action="store_true", help="Skip runs if output_dir/summary.json already exists.")
    parser.add_argument("--ckpt-r101", type=str, default=r"E:/6G/detr-r101-2c7b67e5.pth")
    parser.add_argument("--ckpt-r101-dc5", type=str, default=r"E:/6G/detr-r101-dc5-a2e86def.pth")
    parser.add_argument("--ckpt-r101-panoptic", type=str, default=r"E:/6G/detr-r101-panoptic-40021d53.pth")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / tag
    ensure_dir(run_root)
    manifest_path = run_root / "manifest.jsonl"

    variant_map: Dict[str, Dict[str, Any]] = {
        "v_r50": {
            "variant_id": "v_r50",
            "variant_name": "R50 (hub pretrained)",
            "params": {
                "detr_variant": "detr_resnet50",
                "detr_pretrained": True,
                "detr_checkpoint_path": None,
            },
        },
        "v_r101": {
            "variant_id": "v_r101",
            "variant_name": "R101 (local ckpt)",
            "params": {
                "detr_variant": "detr_resnet101",
                "detr_pretrained": False,
                "detr_checkpoint_path": args.ckpt_r101,
            },
        },
        "v_r101_dc5": {
            "variant_id": "v_r101_dc5",
            "variant_name": "R101-DC5 (local ckpt)",
            "params": {
                "detr_variant": "detr_resnet101_dc5",
                "detr_pretrained": False,
                "detr_checkpoint_path": args.ckpt_r101_dc5,
            },
        },
        "v_r101_panoptic": {
            "variant_id": "v_r101_panoptic",
            "variant_name": "R101-panoptic (local ckpt)",
            "params": {
                "detr_variant": "detr_resnet101_panoptic",
                "detr_pretrained": False,
                "detr_checkpoint_path": args.ckpt_r101_panoptic,
            },
        },
    }
    selected_variant_ids = _parse_str_list(args.variants)
    if args.include_panoptic and "v_r101_panoptic" not in selected_variant_ids:
        selected_variant_ids.append("v_r101_panoptic")

    unknown_variants = sorted(set(selected_variant_ids) - set(variant_map.keys()))
    if unknown_variants:
        raise ValueError(f"Unknown variant ids: {unknown_variants}. Choices: {sorted(variant_map.keys())}")

    variants: List[Dict[str, Any]] = [
        variant_map[v] for v in selected_variant_ids
    ]

    budget_overrides: Dict[str, Any] = {}
    if args.budget_mode == "fast_1h":
        budget_overrides = {
            "max_samples": 2400,
            "epochs": 4,
            "max_train_steps": 70,
            "max_val_steps": 20,
            "batch_size": 20,
            "num_workers": 6,
            "cache_scene_features": True,
            "cache_batch_size": 64,
            "early_stop_patience": 1,
            "early_stop_min_delta": 5e-4,
            "max_wall_time_min": 55.0,
        }

    quick_overrides: Dict[str, Any] = {}
    if args.quick:
        quick_overrides = {
            "max_samples": 512,
            "epochs": 2,
            "max_train_steps": 20,
            "max_val_steps": 10,
        }

    rows: List[Dict[str, Any]] = []
    rid = 0
    for v in variants:
        for seed in seeds:
            rid += 1
            run_id = f"{rid:02d}_{v['variant_id']}_s{seed}"
            out_dir = run_root / run_id
            params = {**v["params"], **budget_overrides, **quick_overrides, "seed": int(seed)}
            summary_path = out_dir / "summary.json"
            if args.resume_skip and summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                summary = _run_one(Path(args.config), out_dir, params)

            row = {
                "run_id": run_id,
                "variant_id": v["variant_id"],
                "variant_name": v["variant_name"],
                "params": params,
                "summary": summary,
                "output_dir": str(out_dir),
            }
            rows.append(row)
            append_jsonl(manifest_path, row)
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    agg = _aggregate(rows)
    write_json(
        run_root / "results.json",
        {
            "tag": tag,
            "seeds": seeds,
            "budget_mode": args.budget_mode,
            "variants": selected_variant_ids,
            "rows": rows,
            "aggregate": agg,
        },
    )

    md = [
        f"# DETR Variant Comparison ({tag})",
        "",
        "Auto-generated by `scripts/run_detr_variant_compare.py`.",
        "",
        "## Aggregate",
        "",
        _md_agg(agg),
        "",
        "## Raw Runs",
        "",
        _md_raw(rows),
    ]
    with open(run_root / "results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    latest = ROOT / "docs" / "detr_variant_compare_latest.md"
    with open(latest, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[Done] Summary JSON: {run_root / 'results.json'}", flush=True)
    print(f"[Done] Summary MD:   {run_root / 'results.md'}", flush=True)
    print(f"[Done] Latest MD:    {latest}", flush=True)


if __name__ == "__main__":
    main()
