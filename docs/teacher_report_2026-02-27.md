# Advisor Report (as of 2026-02-27)

## 1) One-line Goal

Build a reproducible chain:
`DETR query semantics -> multimodal beam decision -> missing-modality robustness -> CARLA closed-loop validation`.

## 2) Current Achievements

### 2.1 Method and engineering

- Implemented `Q-DriveBeam`:
  - Vision branch: DETR query tokens (not global pooling).
  - Comm branches: GPS encoder + Power encoder.
  - Fusion: uncertainty-gated multimodal fusion.
  - Training objectives: main CE + branch CE + branch-fused consistency + gate regularization.
  - Robustness: modality dropout + learnable missing tokens + modality mask path.
- Added local DETR checkpoint support (R101 / R101-DC5 / panoptic).
- Added RTX 4060 acceleration path:
  - CUDA + AMP + TF32 + DataLoader worker tuning.
- Added experiment automation:
  - A6/A7 ablation, A8 DETR variant compare, R1-R4 robustness, full pipeline runner and monitor.

### 2.2 Offline results (latest valid)

#### A6: query pooler ablation (3 seeds)
Source: `outputs/ablation_a6_a7/ccfa_auto_v1_a6/results.md`

| Query Pooler | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---|---:|---:|
| score_weighted_mean | 0.3215 +/- 0.0229 | 3.8210 +/- 0.1099 |
| attn_pool | 0.3181 +/- 0.0151 | 3.8871 +/- 0.1129 |
| cls_cross_attn | **0.3243 +/- 0.0170** | 3.8578 +/- 0.0618 |

Read: `cls_cross_attn` is best on mean Top-1 under current protocol.

#### A7: modality dropout sweep (3 seeds, pooler=cls_cross_attn)
Source: `outputs/ablation_a6_a7/ccfa_auto_v1_a7/results.md`

| modality_dropout_p | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---:|---:|---:|
| 0.0 | 0.3250 +/- 0.0165 | 3.8581 +/- 0.0626 |
| 0.1 | **0.3278 +/- 0.0148** | **3.8545 +/- 0.0797** |
| 0.2 | 0.3215 +/- 0.0105 | 3.8680 +/- 0.1118 |
| 0.3 | 0.3222 +/- 0.0241 | 3.8936 +/- 0.1268 |

Read: `modality_dropout_p=0.1` is currently best on clean-set metrics.

#### R1-R4 robustness stress tests
Source: `outputs/robustness_r1_r4/ccfa_auto_v1_r1r4/results.md`

- baseline: acc=0.3392, loss=3.7988
- R1 (camera blur/occlusion): acc stays in 0.3367~0.3408 (stable).
- R2 (GPS noise 0.5~5m): acc ~0.3392~0.3400 (near-insensitive).
- R3 (power masking): 30% masking drops to 0.3175.
- R4 (missing modality):
  - missing camera: 0.3400
  - missing gps: 0.2758
  - missing power: 0.1550
  - mixed missing: 0.2508

Read: robustness claim is partially supported, but missing-power failure is the main weakness to fix.

### 2.3 A8 DETR variant comparison status

- Pipeline stages `A6 -> A7 -> R1-R4` are complete.
- `A8` was interrupted (`control-C` shown in pipeline log).
- Current `ccfa_auto_v1_a8` status:
  - `detr_resnet50`: 3/3 seeds done, mean test_acc ~ **0.3201**
  - `detr_resnet101`: 2/3 seeds done, mean test_acc ~ **0.3427**
  - missing run: `06_v_r101_s2028`

Resume command:

```bash
cd E:\6G\Code
python scripts/run_detr_variant_compare.py --config configs/scenario36_ccfa_pipeline_gpu.yaml --seeds 2026,2027,2028 --tag ccfa_auto_v1_a8 --resume-skip
```

## 3) Final Technical Route (frozen)

### 3.1 Task structure

- Main task: multimodal beam prediction (offline core contribution).
- Evidence layer: CARLA closed-loop validation.

### 3.2 Model structure

1. DETR query-conditioned representation (replace global pooling).
2. Uncertainty-gated fusion over scene/gps/power.
3. Missing-modality robustness training.
4. Dual consistency:
   - already done: branch-fused consistency loss.
   - pending: simulation-side physical consistency term in CARLA.

### 3.3 Experiment evidence chain

1. Ablation: A0~A8 (A6/A7 done, A8 finishing).
2. Robustness: R1~R4 (done).
3. Closed-loop: C0~C3 in CARLA:
   - C0 baseline run exists.
   - C1/C2/C3 pending.

## 4) Compliance Position vs DETR/MM-MIMO-VI

- We reuse public backbones; we do not claim their original task as our contribution.
- Our contributions are:
  - query-conditioned beam prediction,
  - uncertainty-gated multimodal fusion,
  - missing-modality robustness,
  - offline + closed-loop cross-layer evidence chain.
- Paper should cite original papers/repos and keep license compliance.

## 5) Progress Estimate (honest status)

- Engineering pipeline readiness: about **85%**
- Offline evidence readiness: about **75%**
- Closed-loop evidence readiness: about **30%**
- Overall submission readiness: about **65%**

Interpretation: we now have a coherent and reproducible submission skeleton, but A-level strength still requires full A8 completion and C1-C3 closed-loop evidence.

## 6) Next-week execution plan

1. Finish A8 (complete R101 seed-3) and generate final mean/std comparison table.
2. Strengthen R4 specifically for missing-power robustness.
3. Run CARLA C1 first in advisory mode, then C2/C3.
4. Produce paper table v2: Ablation + Robustness + Closed-loop.
5. Freeze scripts/configs/log index for writing and artifact release.
