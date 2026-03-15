# Q-DriveBeam (DETR + Multimodal Fusion + Consistency)

Clean research codebase for integrating DETR scene understanding into 6G beam prediction.

This project is designed for:
- fusion (not copying DETR/MM-MIMO-VI as-is),
- reproducible experiments on `scenario36`,
- clean extension to autonomous-driving simulation demos (CARLA).

## What is implemented

- Scenario36 dataset adapter (`gps + power + rgb`).
- Pretrained DETR query-token extractor (from `E:\6G\detr-main` by default).
- Uncertainty-aware modality gating.
- Fused/branch consistency regularization.
- Single-entry train pipeline with checkpointing and JSON logs.
- CARLA-friendly inference adapter (`src/beamfusion/carla_adapter.py`).

## Quick Start

```bash
cd E:\6G\Code
python -m pip install -r requirements.txt
python scripts/train_scenario36.py --config configs/scenario36_detr_fusion.yaml
```

4060 GPU preset:

```bash
python scripts/train_scenario36.py --config configs/scenario36_gpu_4060.yaml
```

GPU smoke check:

```bash
python scripts/train_scenario36.py --config configs/smoke_cuda.yaml
```

One-command CCF-A pipeline (A6 -> A7 -> A8, multi-seed, resume enabled):

```bash
python scripts/run_ccfa_pipeline.py --config configs/scenario36_ccfa_pipeline_gpu.yaml --seeds 2026,2027,2028 --tag-prefix ccfa_auto_v1
```

One-hour screening pipeline on RTX 4060 (fast budget, resume enabled):

```bash
python scripts/run_ccfa_pipeline.py --config configs/scenario36_fast_1h_4060.yaml --budget-mode fast_1h --seeds 2026,2027,2028 --tag-prefix ccfa_fast1h_v1
```

Run A6/A7 ablations (auto logs + markdown summary):

```bash
python scripts/run_ablation_a6_a7.py --mode both --quick
```

Multi-seed paper-ready ablation (example):

```bash
python scripts/run_ablation_a6_a7.py --mode both --seeds 2026,2027,2028 --tag a6a7_full_s123
```

DETR backbone/checkpoint comparison (A8):

```bash
python scripts/run_detr_variant_compare.py --quick --seeds 2026,2027 --include-panoptic
```

Resume/skip finished runs:

```bash
python scripts/run_ablation_a6_a7.py --mode a6 --seeds 2026,2027,2028 --resume-skip --tag full_a6_s123
```

Top-1 expert baseline (power+log, same split, strong reference):

```bash
python scripts/run_top1_hgb_expert.py --power-use-log --tag top1_full_20260301
```

Fast budget (single-stage runs, <=1h cap per run):

```bash
python scripts/run_ablation_a6_a7.py --mode both --budget-mode fast_1h --seeds 2026 --tag fast1h_a6a7_v1
python scripts/run_detr_variant_compare.py --budget-mode fast_1h --variants v_r50,v_r101 --seeds 2026 --tag fast1h_a8_v1
```

Monitor an active run directory:

```bash
python scripts/monitor_experiment.py --run-root E:/6G/Code/outputs/ablation_a6_a7/full_a6_s2027_2028_gpu
```

Use local DETR checkpoints (example via inline override):

```bash
python -c "from beamfusion.config import load_config; from beamfusion.train import train; \
cfg=load_config('E:/6G/Code/configs/smoke.yaml', override={ \
'detr_variant':'detr_resnet101_dc5', \
'detr_pretrained':False, \
'detr_checkpoint_path':'E:/6G/detr-r101-dc5-a2e86def.pth'}); \
print(train(cfg))"
```

## Project Layout

```text
configs/
docs/
scripts/
src/beamfusion/
```

## Notes

- DETR uses pretrained weights by default (no from-scratch COCO training required).
- The current default is CPU-safe but slow; use GPU for real experiments.
- For frozen DETR runs, enable `cache_scene_features=true` to avoid repeated DETR forward every epoch.
- Each run writes debug-friendly artifacts (`config.json`, `run_meta.json`, `train_log.jsonl`, `summary.json`).
- CUDA acceleration requires a CUDA-enabled PyTorch build (`torch.cuda.is_available()` should be `True`).
