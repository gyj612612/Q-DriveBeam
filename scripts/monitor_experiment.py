from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor progress of a run directory.")
    parser.add_argument("--run-root", type=str, required=True, help="e.g. E:/6G/Code/outputs/ablation_a6_a7/full_a6_s2027_2028_gpu")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest = _load_manifest(run_root / "manifest.jsonl")
    summaries = list(run_root.rglob("summary.json"))
    train_logs = list(run_root.rglob("train_log.jsonl"))

    print(f"run_root={run_root}")
    print(f"manifest_rows={len(manifest)}")
    print(f"completed_summaries={len(summaries)}")
    print(f"train_logs={len(train_logs)}")
    if manifest:
        print("last_manifest_row:")
        print(json.dumps(manifest[-1], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

