# BeamFusion Code Explanation (Living Doc)

Last updated: 2026-02-26

## 1) Project Goal

Build a reproducible multimodal beam predictor that integrates:
- DETR query-conditioned visual representation,
- uncertainty-gated fusion for communication modalities,
- robustness under missing modalities.

This document is a living note and should be updated whenever model/training logic changes.

## 2) Core Modules

- `src/beamfusion/data/scenario36.py`
  - Loads scenario36 pickle data.
  - Builds train/val/test split with fixed seed.
  - Normalizes GPS/Power using train statistics.
  - Optional power log feature augmentation via `power_use_log=true`.

- `src/beamfusion/models/detr_tokens.py`
  - Loads pretrained DETR (local repo or torch hub).
  - Can optionally load a local checkpoint via `detr_checkpoint_path`.
  - Extracts top-K query candidates by objectness.
  - Supports query pooling modes:
    - `score_weighted_mean` (baseline),
    - `attn_pool`,
    - `cls_cross_attn`.

- `src/beamfusion/models/fusion.py`
  - Uncertainty-gated modality fusion.
  - Supports optional `modality_mask` to ignore dropped modalities.

- `src/beamfusion/models/model.py`
  - Full model wrapper: scene/gps/power encoders + fusion head.
  - Adds train-time modality dropout with learnable missing tokens.
  - Supports optional branch denoising AE on GPS/Power (`ae_enabled`).
  - Exposes scene pooling weights for analysis/debug.

- `src/beamfusion/losses.py`
  - Main CE loss + branch CE + symmetric consistency KL + gate regularization.
  - Optional IEMF-style inverse-effect coefficient (`iemf_enabled`).
  - Optional AE reconstruction/KL terms (`ae_recon_lambda`, `ae_kl_lambda`).

- `src/beamfusion/train.py`
  - End-to-end train/eval pipeline.
  - Saves checkpoints and detailed json/jsonl logs.
  - Supports CUDA acceleration knobs: AMP, TF32, data-loader worker tuning.
  - Reports Top-1/Top-3/Top-5 for train/val/test.

## 3) Logging and Debug Files

Each run writes into `outputs/<exp_name>/`:

- `config.json`: resolved config used in this run.
- `run_meta.json`: run start info (time/seed/device).
- `train_log.jsonl`: per-epoch train/val metrics.
  - Includes: `loss`, `acc`, `main`, `branch`, `consistency`, `gate_reg`.
- `checkpoints/best.pt`: best checkpoint by validation accuracy.
- `summary.json`: final summary with best epoch and test metrics.

These files are designed to support post-mortem debugging without rerunning experiments.

## 4) A6 and A7 Ablations

- `A6`: Query pooler ablation
  - Compare `score_weighted_mean`, `attn_pool`, `cls_cross_attn`.
  - Goal: verify query-conditioned design gain.

- `A7`: Missing-modality training ablation
  - Sweep `modality_dropout_p` (e.g. `0.0/0.1/0.2/0.3`).
  - Goal: verify robustness gain under modality failure.

Auto-run script:
- `scripts/run_ablation_a6_a7.py`
- `scripts/run_detr_variant_compare.py` (A8 backbone/checkpoint comparison)

Outputs:
- `outputs/ablation_a6_a7/<tag>/manifest.jsonl`
- `outputs/ablation_a6_a7/<tag>/results.json`
- `outputs/ablation_a6_a7/<tag>/results.md`
- `docs/ablation_a6_a7_latest.md` (latest auto summary)

## 5) Reproducible Commands

Single training run:

```bash
python scripts/train_scenario36.py --config configs/scenario36_detr_fusion.yaml
```

A6+A7 quick debug run:

```bash
python scripts/run_ablation_a6_a7.py --mode both --quick
```

Production ablation run:

```bash
python scripts/run_ablation_a6_a7.py --mode both --tag prod_v1
```

4060 GPU preset run:

```bash
python scripts/train_scenario36.py --config configs/scenario36_gpu_4060.yaml
```
