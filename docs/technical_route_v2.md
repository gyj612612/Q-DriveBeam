# Technical Route v2 (CCF-A Oriented)

Date: 2026-02-26

## 1) One-line Objective

Build a robust multimodal beam prediction system where DETR query tokens guide communication beam decisions, and validate benefits in both offline beam benchmarks and CARLA closed-loop simulation.

## 2) Final Method Stack

Model name (working): `Q-DriveBeam`.

- Visual branch:
  - Pretrained DETR (`detr-main` local or local `.pth` checkpoint).
  - Query token extraction + configurable query pooler.
- Communication branches:
  - GPS encoder.
  - Power encoder.
- Fusion:
  - Uncertainty-gated fusion with optional modality mask.
- Training objectives:
  - Main CE.
  - Branch auxiliary CE.
  - Fused-vs-branch consistency.
  - Gate regularization.
- Robustness:
  - Train-time modality dropout with learnable missing tokens.

## 3) Key Scientific Claims

- Claim C1 (Query-conditioned):
  Query-level scene representations improve beam prediction over plain pooled visual features.
- Claim C2 (Dual consistency):
  Consistency constraints stabilize multimodal training and improve generalization.
- Claim C3 (Robustness):
  Missing-modality training plus uncertainty gate reduces performance collapse under sensor corruption/failure.
- Claim C4 (Transfer to driving):
  Beam-aware guidance helps closed-loop autonomous driving quality under realistic disturbances.

## 4) Experiment Evidence Chain

Offline:
- A0-A7 ablations.
- R1-R4 robustness stresses.
- 3 seeds minimum with mean/std.

Closed-loop CARLA:
- C0-C3 protocol.
- Metrics: collisions, route completion, average speed, stuck ratio, FPS.

Reproducibility:
- one config per run,
- run metadata + train logs + summary in run directory,
- aggregate scripts generate markdown/json tables.

## 5) What You Need To Download (Current Stage)

Not required now:
- COCO train/val datasets.
- DETR from-scratch training resources.

Already sufficient for current route:
- `E:\6G\detr-main` repo.
- Local DETR checkpoints:
  - `E:\6G\detr-r101-dc5-a2e86def.pth`
  - `E:\6G\detr-r101-panoptic-40021d53.pth`
- Your scenario36/DeepSense data already used in pipeline.

Optional later (only if needed):
- Additional DETR checkpoint `detr-r101-2c7b67e5.pth` for a non-DC5 R101 comparison.
- Panoptic API dependencies only when running panoptic visualization notebooks.

## 6) Current Risks and Mitigations

- Risk: apparent gains come from training budget differences.
  - Mitigation: fixed budget and seeds for all comparisons.
- Risk: modality dropout hurts clean-set performance.
  - Mitigation: evaluate robustness under R4, not clean set alone.
- Risk: CARLA effects are noisy.
  - Mitigation: fixed scenario seeds and repeated runs.

## 7) Immediate Next Milestones

1. Run full-budget A6 with seeds `2026,2027,2028`.
2. Select best query pooler from aggregate metrics.
3. Run full-budget A7 (modality dropout sweep) with chosen pooler.
4. Start R1-R4 stress bench.
5. Integrate C1 advisory mode in CARLA and produce C0/C1 tables.

## 8) Execution Status Snapshot (Route Alignment)

This project is following the agreed route:
`DETR query semantics -> beam decision -> missing-modality robustness -> CARLA closed-loop evidence`.

Status by stage:
- V1 Query-conditioned module:
  - Status: implemented and validated.
  - Evidence: A6 quick single-seed and 2-seed runs completed.
- V2 Missing-modality robustness:
  - Status: implemented (modality dropout + missing token + uncertainty mask path), early sanity runs done.
  - Evidence: A7 quick runs completed; full-budget robustness sweeps pending.
- V3 Dual consistency:
  - Status: model consistency is implemented; simulator/physics consistency term is pending.
  - Evidence: current `losses.py` includes branch-fused consistency, CARLA-side physics consistency not yet added.
- CARLA validation layer:
  - Status: baseline and sandbox are ready; BeamFusion advisory integration is next.
  - Evidence: traditional CARLA run and metrics logged; C1-C3 experiment line pending.

Current route-completion estimate:
- Engineering pipeline readiness: ~75%.
- Paper-evidence readiness (for A submission): ~40%.
- Overall route completion: ~50%.

Acceleration update (2026-02-26):
- CUDA-enabled PyTorch is active on RTX 4060 (`torch 2.4.1+cu124`).
- Training pipeline now supports AMP + TF32 + worker tuning.
- Added `configs/scenario36_gpu_4060.yaml` and `configs/smoke_cuda.yaml`.

## 9) Runtime Protocol Update (2026-03-01)

To control iteration latency on RTX 4060, we now use a two-stage runtime protocol:

Stage S1 (Fast-1h screening):
- Target: complete one comparison cycle within about one hour.
- Key switches:
  - `cache_scene_features=true` when `freeze_detr=true`,
  - `early_stop_patience` enabled,
  - `max_wall_time_min` hard cap,
  - pipeline `--budget-mode fast_1h`.
- Config:
  - `configs/scenario36_fast_1h_4060.yaml`.

Stage S2 (Paper-budget confirmation):
- Only run on shortlisted winners from S1.
- Multi-seed and full table generation for submission claims.

New script features:
- `run_ablation_a6_a7.py`:
  - `--budget-mode fast_1h`,
  - `--a6-pool-modes`,
  - `--a7-drop-values`.
- `run_detr_variant_compare.py`:
  - `--budget-mode fast_1h`,
  - `--variants` (e.g., `v_r50,v_r101`).
- `run_ccfa_pipeline.py`:
  - `--budget-mode {paper,fast_1h}` for end-to-end fast screening.
