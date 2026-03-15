from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "outputs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n[{_now()}] RUN {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=log,
            text=True,
        )
        return p.wait()


def _update_live(status_path: Path, stage: str, state: str, extra: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "time_utc": _now(),
        "stage": stage,
        "state": state,
    }
    if extra:
        payload.update(extra)
    prev = {}
    if status_path.exists():
        prev = _read_json(status_path)
    prev.setdefault("events", [])
    prev["events"].append(payload)
    prev["last"] = payload
    _write_json(status_path, prev)


def _best_pool_from_a6(a6_results_json: Path) -> str:
    data = _read_json(a6_results_json)
    agg = data.get("aggregate", [])
    if not agg:
        raise RuntimeError(f"No aggregate rows in {a6_results_json}")
    best = sorted(
        agg,
        key=lambda r: (float(r["test_acc_mean"]), -float(r["test_loss_mean"])),
        reverse=True,
    )[0]
    return str(best["query_pool_mode"])


def _build_report(
    path: Path,
    a6_json: Path,
    a7_json: Path,
    a8_json: Path,
    r_json: Path,
    best_pool: str,
    started: str,
) -> None:
    a6 = _read_json(a6_json)
    a7 = _read_json(a7_json)
    a8 = _read_json(a8_json)
    rr = _read_json(r_json)
    ended = _now()

    def table_from_agg(rows: List[Dict[str, Any]], tag: str) -> List[str]:
        lines = [
            f"### {tag}",
            "",
            "| setting | test_top1 (mean+/-std) | test_top3 (mean+/-std) | test_top5 (mean+/-std) | test_loss (mean+/-std) |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in rows:
            if "query_pool_mode" in r:
                name = f"pool={r['query_pool_mode']}, drop={float(r.get('modality_dropout_p', 0.0)):.2f}"
            else:
                name = str(r.get("variant_name", r.get("variant_id", "unknown")))
            lines.append(
                "| {name} | {t1m:.4f} +/- {t1s:.4f} | {t3m:.4f} +/- {t3s:.4f} | {t5m:.4f} +/- {t5s:.4f} | {lossm:.4f} +/- {losss:.4f} |".format(
                    name=name,
                    t1m=float(r.get("test_top1_mean", r.get("test_acc_mean", 0.0))),
                    t1s=float(r.get("test_top1_std", r.get("test_acc_std", 0.0))),
                    t3m=float(r.get("test_top3_mean", r.get("test_acc_mean", 0.0))),
                    t3s=float(r.get("test_top3_std", r.get("test_acc_std", 0.0))),
                    t5m=float(r.get("test_top5_mean", r.get("test_acc_mean", 0.0))),
                    t5s=float(r.get("test_top5_std", r.get("test_acc_std", 0.0))),
                    lossm=float(r["test_loss_mean"]),
                    losss=float(r["test_loss_std"]),
                )
            )
        lines.append("")
        return lines

    md: List[str] = [
        "# CCF-A Pipeline Auto Report",
        "",
        f"- start_utc: {started}",
        f"- end_utc: {ended}",
        f"- selected_best_pool_from_A6: `{best_pool}`",
        "",
    ]
    md.extend(table_from_agg(a6.get("aggregate", []), "A6 (Query Pooler)"))
    md.extend(table_from_agg(a7.get("aggregate", []), "A7 (Missing Modality)"))
    md.extend(table_from_agg(a8.get("aggregate", []), "A8 (DETR Variant)"))
    md.extend(
        [
            "### R1-R4 (Robustness Cases)",
            "",
            "| case_id | group | acc | loss |",
            "|---|---|---:|---:|",
        ]
    )
    for r in rr.get("rows", []):
        md.append(f"| {r.get('case_id','')} | {r.get('group','')} | {float(r['acc']):.4f} | {float(r['loss']):.4f} |")
    md.append("")
    md.extend(
        [
            "## Artifacts",
            "",
            f"- A6: `{a6_json}`",
            f"- A7: `{a7_json}`",
            f"- A8: `{a8_json}`",
            f"- R1-R4: `{r_json}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end CCF-A experiment pipeline with resume.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "scenario36_detr_fusion_gpu.yaml"))
    parser.add_argument("--seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--include-panoptic", action="store_true")
    parser.add_argument("--tag-prefix", type=str, default="ccfa_v1")
    parser.add_argument("--robust-max-steps", type=int, default=80)
    parser.add_argument("--budget-mode", type=str, default="paper", choices=["paper", "fast_1h"])
    args = parser.parse_args()

    started = _now()
    status_path = OUT / "ccfa_pipeline" / f"{args.tag_prefix}_live_status.json"
    pipeline_log = OUT / "ccfa_pipeline" / f"{args.tag_prefix}_pipeline.log"

    tags = {
        "a6": f"{args.tag_prefix}_a6",
        "a7": f"{args.tag_prefix}_a7",
        "r": f"{args.tag_prefix}_r1r4",
        "a8": f"{args.tag_prefix}_a8",
    }
    seeds = args.seeds
    a6_pool_modes = "score_weighted_mean,attn_pool,cls_cross_attn"
    a7_drop_values = "0.0,0.1,0.2,0.3"
    a8_variants = "v_r50,v_r101,v_r101_dc5"
    robust_steps = int(args.robust_max_steps)

    if args.budget_mode == "fast_1h":
        # Fast screening profile intended to finish one full comparison cycle within ~1 hour on RTX 4060.
        first_seed = [x.strip() for x in args.seeds.split(",") if x.strip()][:1]
        seeds = ",".join(first_seed or ["2026"])
        a6_pool_modes = "score_weighted_mean,cls_cross_attn"
        a7_drop_values = "0.0,0.1,0.2"
        a8_variants = "v_r50,v_r101"
        robust_steps = min(int(args.robust_max_steps), 30)

    # Stage A6
    _update_live(status_path, "A6", "start", {"tag": tags["a6"], "budget_mode": args.budget_mode})
    cmd_a6 = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_ablation_a6_a7.py"),
        "--config",
        args.config,
        "--mode",
        "a6",
        "--a6-pool-modes",
        a6_pool_modes,
        "--budget-mode",
        "fast_1h" if args.budget_mode == "fast_1h" else "none",
        "--seeds",
        seeds,
        "--tag",
        tags["a6"],
        "--resume-skip",
    ]
    rc = _run(cmd_a6, pipeline_log)
    if rc != 0:
        _update_live(status_path, "A6", "failed", {"return_code": rc})
        raise SystemExit(rc)
    _update_live(status_path, "A6", "done")

    a6_json = OUT / "ablation_a6_a7" / tags["a6"] / "results.json"
    best_pool = _best_pool_from_a6(a6_json)
    _update_live(status_path, "A6", "best_pool_selected", {"best_pool": best_pool})

    # Stage A7
    _update_live(status_path, "A7", "start", {"tag": tags["a7"], "best_pool": best_pool})
    cmd_a7 = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_ablation_a6_a7.py"),
        "--config",
        args.config,
        "--mode",
        "a7",
        "--a7-base-pool",
        best_pool,
        "--a7-drop-values",
        a7_drop_values,
        "--budget-mode",
        "fast_1h" if args.budget_mode == "fast_1h" else "none",
        "--seeds",
        seeds,
        "--tag",
        tags["a7"],
        "--resume-skip",
    ]
    rc = _run(cmd_a7, pipeline_log)
    if rc != 0:
        _update_live(status_path, "A7", "failed", {"return_code": rc})
        raise SystemExit(rc)
    _update_live(status_path, "A7", "done")

    # Stage R1-R4 robustness
    _update_live(status_path, "R1-R4", "start", {"tag": tags["r"]})
    cmd_r = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_robustness_r1_r4.py"),
        "--a7-results-json",
        str(OUT / "ablation_a6_a7" / tags["a7"] / "results.json"),
        "--tag",
        tags["r"],
        "--max-steps",
        str(robust_steps),
    ]
    rc = _run(cmd_r, pipeline_log)
    if rc != 0:
        _update_live(status_path, "R1-R4", "failed", {"return_code": rc})
        raise SystemExit(rc)
    _update_live(status_path, "R1-R4", "done")

    # Stage A8
    _update_live(status_path, "A8", "start", {"tag": tags["a8"]})
    cmd_a8 = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_detr_variant_compare.py"),
        "--config",
        args.config,
        "--variants",
        a8_variants,
        "--budget-mode",
        "fast_1h" if args.budget_mode == "fast_1h" else "none",
        "--seeds",
        seeds,
        "--tag",
        tags["a8"],
        "--resume-skip",
    ]
    if args.include_panoptic:
        cmd_a8.append("--include-panoptic")
    rc = _run(cmd_a8, pipeline_log)
    if rc != 0:
        _update_live(status_path, "A8", "failed", {"return_code": rc})
        raise SystemExit(rc)
    _update_live(status_path, "A8", "done")

    # Final report
    report = DOCS / "ccfa_pipeline_report_auto.md"
    _build_report(
        path=report,
        a6_json=OUT / "ablation_a6_a7" / tags["a6"] / "results.json",
        a7_json=OUT / "ablation_a6_a7" / tags["a7"] / "results.json",
        a8_json=OUT / "detr_variant_compare" / tags["a8"] / "results.json",
        r_json=OUT / "robustness_r1_r4" / tags["r"] / "results.json",
        best_pool=best_pool,
        started=started,
    )
    _update_live(status_path, "pipeline", "done", {"report": str(report)})


if __name__ == "__main__":
    main()
