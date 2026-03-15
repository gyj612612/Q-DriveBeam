from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beamfusion.config import load_config
from beamfusion.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DETR-fusion beam predictor on scenario36.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "scenario36_detr_fusion.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = train(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
