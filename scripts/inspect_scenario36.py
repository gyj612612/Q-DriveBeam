from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beamfusion.data import prepare_scenario36


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=str, default=r"E:/6G/scenario36_merged")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    data = prepare_scenario36(
        scenario_root=args.scenario_root,
        seed=2026,
        train_ratio=0.6,
        val_ratio=0.2,
        image_key_a="unit1_rgb5",
        image_key_b="unit1_rgb6",
        max_samples=args.max_samples,
    )
    summary = {
        "gps_dim": int(data.gps.shape[1]),
        "power_dim": int(data.power.shape[1]),
        "num_samples": int(len(data.labels)),
        "num_classes": int(data.labels.max() + 1),
        "train_size": int(len(data.train_idx)),
        "val_size": int(len(data.val_idx)),
        "test_size": int(len(data.test_idx)),
        "scenario_dir": str(data.scenario_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
