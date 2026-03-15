from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beamfusion.data import prepare_scenario36
from beamfusion.utils import ensure_dir, write_json


def _score_topk(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    labels = np.arange(proba.shape[1])
    return float(top_k_accuracy_score(y_true, proba, k=k, labels=labels))


def _candidate_grid() -> List[Dict[str, Any]]:
    return [
        {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "l2_regularization": 1e-3,
        },
        {
            "max_iter": 500,
            "learning_rate": 0.05,
            "max_depth": 12,
            "min_samples_leaf": 20,
            "l2_regularization": 1e-3,
        },
        {
            "max_iter": 500,
            "learning_rate": 0.05,
            "max_depth": 14,
            "min_samples_leaf": 15,
            "l2_regularization": 5e-4,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-1 focused HGB expert on scenario36 power features.")
    parser.add_argument("--scenario-root", type=str, default="E:/6G/scenario36_merged")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--power-use-log", action="store_true")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs" / "top1_hgb_expert"))
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    tag = args.tag.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / tag
    ensure_dir(out_dir)

    prepared = prepare_scenario36(
        scenario_root=args.scenario_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        image_key_a="unit1_rgb5",
        image_key_b="unit1_rgb6",
        power_use_log=bool(args.power_use_log),
        power_log_clip_min=1e-6,
        max_samples=args.max_samples,
    )

    x = prepared.power
    y = prepared.labels
    tr = prepared.train_idx
    va = prepared.val_idx
    te = prepared.test_idx

    candidates = _candidate_grid()
    rows: List[Dict[str, Any]] = []
    best_score = -1.0
    best_cfg: Dict[str, Any] | None = None
    best_clf: HistGradientBoostingClassifier | None = None
    for i, cfg in enumerate(candidates, start=1):
        clf = HistGradientBoostingClassifier(random_state=args.seed, **cfg)
        clf.fit(x[tr], y[tr])
        p_va = clf.predict_proba(x[va])
        val_top1 = _score_topk(y[va], p_va, k=1)
        row = {"id": i, "cfg": cfg, "val_top1": val_top1}
        rows.append(row)
        if val_top1 > best_score:
            best_score = val_top1
            best_cfg = cfg
            best_clf = clf

    assert best_clf is not None and best_cfg is not None
    p_te = best_clf.predict_proba(x[te])
    pred_te = np.argmax(p_te, axis=1)

    summary = {
        "tag": tag,
        "seed": int(args.seed),
        "scenario_root": args.scenario_root,
        "power_use_log": bool(args.power_use_log),
        "max_samples": args.max_samples,
        "num_samples": int(len(y)),
        "train_size": int(len(tr)),
        "val_size": int(len(va)),
        "test_size": int(len(te)),
        "num_classes": int(y.max() + 1),
        "best_cfg": best_cfg,
        "best_val_top1": float(best_score),
        "test_top1": float(accuracy_score(y[te], pred_te)),
        "test_top3": _score_topk(y[te], p_te, k=3),
        "test_top5": _score_topk(y[te], p_te, k=5),
        "candidates": rows,
    }
    write_json(out_dir / "summary.json", summary)

    md = [
        f"# Top-1 HGB Expert ({tag})",
        "",
        f"- seed: {args.seed}",
        f"- power_use_log: {bool(args.power_use_log)}",
        f"- samples: {len(y)}",
        "",
        "## Best Result",
        "",
        f"- best_val_top1: **{summary['best_val_top1']:.4f}**",
        f"- test_top1: **{summary['test_top1']:.4f}**",
        f"- test_top3: **{summary['test_top3']:.4f}**",
        f"- test_top5: **{summary['test_top5']:.4f}**",
        "",
        "## Candidate Grid",
        "",
        "| id | max_iter | lr | max_depth | min_samples_leaf | l2 | val_top1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        c = r["cfg"]
        md.append(
            f"| {r['id']} | {c['max_iter']} | {c['learning_rate']:.3f} | {c['max_depth']} | {c['min_samples_leaf']} | {c['l2_regularization']:.4f} | {r['val_top1']:.4f} |"
        )

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latest = ROOT / "docs" / "top1_hgb_expert_latest.md"
    latest.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Done] summary: {out_dir / 'summary.json'}")
    print(f"[Done] latest:  {latest}")


if __name__ == "__main__":
    main()
