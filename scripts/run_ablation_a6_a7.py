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
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("No valid seeds parsed from --seeds.")
    return out


def _parse_str_list(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Parsed empty list.")
    return out


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("Parsed empty float list.")
    return out


def _md_table_raw(rows: Sequence[Dict[str, Any]]) -> str:
    head = "| Run | Group | Seed | query_pool_mode | modality_dropout_p | best_val_acc | test_top1 | test_top3 | test_top5 | test_loss | output_dir |\n"
    sep = "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n"
    body = []
    for r in rows:
        s = r["summary"]
        p = r["params"]
        body.append(
            "| {id} | {group} | {seed} | {pool} | {drop:.2f} | {val:.4f} | {top1:.4f} | {top3:.4f} | {top5:.4f} | {loss:.4f} | `{out}` |".format(
                id=r["run_id"],
                group=r["group"],
                seed=int(p["seed"]),
                pool=p.get("query_pool_mode", ""),
                drop=float(p.get("modality_dropout_p", 0.0)),
                val=float(s["best_val_acc"]),
                top1=float(s.get("test_top1", s["test_acc"])),
                top3=float(s.get("test_top3", s["test_acc"])),
                top5=float(s.get("test_top5", s["test_acc"])),
                loss=float(s["test_loss"]),
                out=r["output_dir"],
            )
        )
    return head + sep + "\n".join(body) + "\n"


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        p = r["params"]
        key = f"{r['group']}|{p.get('query_pool_mode')}|{float(p.get('modality_dropout_p', 0.0)):.2f}"
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        first = rs[0]
        val = [float(r["summary"]["best_val_acc"]) for r in rs]
        top1 = [float(r["summary"].get("test_top1", r["summary"]["test_acc"])) for r in rs]
        top3 = [float(r["summary"].get("test_top3", r["summary"]["test_acc"])) for r in rs]
        top5 = [float(r["summary"].get("test_top5", r["summary"]["test_acc"])) for r in rs]
        loss = [float(r["summary"]["test_loss"]) for r in rs]
        out.append(
            {
                "group": first["group"],
                "query_pool_mode": first["params"].get("query_pool_mode"),
                "modality_dropout_p": float(first["params"].get("modality_dropout_p", 0.0)),
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


def _md_table_agg(rows: Sequence[Dict[str, Any]]) -> str:
    head = "| Group | query_pool_mode | modality_dropout_p | num_seeds | best_val_acc (mean+/-std) | test_top1 (mean+/-std) | test_top3 (mean+/-std) | test_top5 (mean+/-std) | test_loss (mean+/-std) |\n"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for r in rows:
        body.append(
            "| {group} | {pool} | {drop:.2f} | {n} | {vmean:.4f} +/- {vstd:.4f} | {t1m:.4f} +/- {t1s:.4f} | {t3m:.4f} +/- {t3s:.4f} | {t5m:.4f} +/- {t5s:.4f} | {lmean:.4f} +/- {lstd:.4f} |".format(
                group=r["group"],
                pool=r["query_pool_mode"],
                drop=float(r["modality_dropout_p"]),
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


def _run_one(config_path: Path, output_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config(str(config_path), override={**overrides, "output_dir": str(output_dir)})
    summary = train(cfg)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A6/A7 ablations with automatic logs and markdown summary.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "scenario36_detr_fusion.yaml"))
    parser.add_argument("--mode", type=str, default="both", choices=["a6", "a7", "both"])
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs" / "ablation_a6_a7"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--a7-base-pool", type=str, default="attn_pool")
    parser.add_argument(
        "--a6-pool-modes",
        type=str,
        default="score_weighted_mean,attn_pool,cls_cross_attn",
        help="Comma-separated list for A6, e.g. score_weighted_mean,cls_cross_attn",
    )
    parser.add_argument(
        "--a7-drop-values",
        type=str,
        default="0.0,0.1,0.2,0.3",
        help="Comma-separated dropout values for A7, e.g. 0.0,0.1,0.2",
    )
    parser.add_argument("--seeds", type=str, default="2026", help="Comma-separated seeds, e.g. 2026,2027,2028")
    parser.add_argument("--budget-mode", type=str, default="none", choices=["none", "fast_1h"])
    parser.add_argument("--quick", action="store_true", help="Use small train/val steps for debug.")
    parser.add_argument("--resume-skip", action="store_true", help="Skip runs if output_dir/summary.json already exists.")
    args = parser.parse_args()

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / tag
    ensure_dir(run_root)
    manifest_path = run_root / "manifest.jsonl"
    seeds = _parse_seeds(args.seeds)

    tasks: List[Dict[str, Any]] = []
    if args.mode in {"a6", "both"}:
        for pool_mode in _parse_str_list(args.a6_pool_modes):
            tasks.append(
                {
                    "group": "A6",
                    "name": pool_mode,
                    "params": {
                        "query_pool_mode": pool_mode,
                        "modality_dropout_p": 0.0,
                    },
                }
            )
    if args.mode in {"a7", "both"}:
        for p in _parse_float_list(args.a7_drop_values):
            tasks.append(
                {
                    "group": "A7",
                    "name": f"drop_{p:.1f}",
                    "params": {
                        "query_pool_mode": args.a7_base_pool,
                        "modality_dropout_p": p,
                    },
                }
            )

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

    if args.quick:
        quick_overrides = {
            "max_samples": 512,
            "epochs": 2,
            "max_train_steps": 20,
            "max_val_steps": 10,
        }
    else:
        quick_overrides = {}

    rows: List[Dict[str, Any]] = []
    rid = 0
    for t in tasks:
        for seed in seeds:
            rid += 1
            run_id = f"{rid:02d}_{t['group'].lower()}_{t['name']}_s{seed}"
            out_dir = run_root / run_id
            params = {**t["params"], **budget_overrides, **quick_overrides, "seed": int(seed)}
            summary_path = out_dir / "summary.json"
            if args.resume_skip and summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                summary = _run_one(Path(args.config), out_dir, params)

            row = {
                "run_id": run_id,
                "group": t["group"],
                "params": params,
                "summary": summary,
                "output_dir": str(out_dir),
            }
            rows.append(row)
            append_jsonl(manifest_path, row)
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    agg_rows = _aggregate_rows(rows)
    write_json(
        run_root / "results.json",
        {
            "tag": tag,
            "seeds": seeds,
            "budget_mode": args.budget_mode,
            "a6_pool_modes": _parse_str_list(args.a6_pool_modes),
            "a7_drop_values": _parse_float_list(args.a7_drop_values),
            "rows": rows,
            "aggregate": agg_rows,
        },
    )

    md = [
        f"# A6/A7 Ablation Results ({tag})",
        "",
        "This file is auto-generated by `scripts/run_ablation_a6_a7.py`.",
        "",
        "## Aggregate (for paper)",
        "",
        _md_table_agg(agg_rows),
        "",
        "## Raw Runs",
        "",
        _md_table_raw(rows),
    ]
    with open(run_root / "results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    latest = ROOT / "docs" / "ablation_a6_a7_latest.md"
    with open(latest, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[Done] Summary JSON: {run_root / 'results.json'}", flush=True)
    print(f"[Done] Summary MD:   {run_root / 'results.md'}", flush=True)
    print(f"[Done] Latest MD:    {latest}", flush=True)


if __name__ == "__main__":
    main()
