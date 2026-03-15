# Paper Experiment Matrix v1 (CCF-A Oriented)

Date: 2026-02-26

## 1) Offline Beam Prediction (Scenario36)

Primary metrics:
- Top-1 accuracy
- Top-5 accuracy
- mAP / ROC-AUC
- Calibration (ECE)

### A. Ablation (core method validity)

| ID | Setting | Purpose |
|---|---|---|
| A0 | Full model: DETR query-conditioned + uncertainty gate + dual consistency + modality dropout | Main claim |
| A1 | Replace DETR query tokens with pooled RGB feature | Verify query-level contribution |
| A2 | Remove dual consistency loss | Verify consistency contribution |
| A3 | Remove uncertainty gate (uniform/static fusion) | Verify robust fusion contribution |
| A4 | Single-view camera (no dual view) | Verify multi-view contribution |
| A5 | Freeze-all DETR vs partial fine-tune | Verify transfer strategy |
| A6 | Query pooler: `score_weighted_mean` vs `attn_pool` vs `cls_cross_attn` | Verify query-conditioned design |
| A7 | Modality-drop train off vs on (`modality_dropout_p`) | Verify missing-modality training contribution |
| A8 | DETR variant/pretrain compare: R50 vs R101 vs R101-DC5 (optional panoptic pretrain) | Verify backbone/pretrain transfer effect |

### B. Robustness (real deployment relevance)

| ID | Stress Type | Sweep |
|---|---|---|
| R1 | Camera blur/occlusion | blur sigma {1,2,3}, occlusion {10%,20%,30%} |
| R2 | GPS noise | std {0.5m, 1m, 2m, 5m} |
| R3 | Power feature corruption | masking ratio {10%,20%,30%} |
| R4 | Missing modality | drop camera / gps / power / mixed |

## 2) Closed-loop CARLA Evaluation

Primary metrics:
- Collision count
- Route completion rate
- Average speed
- Stuck ratio (time with very low speed)
- Approx FPS

### C. Closed-loop comparative settings

| ID | Setting | Purpose |
|---|---|---|
| C0 | Traditional controller baseline (existing script) | Baseline line |
| C1 | C0 + BeamFusion recommendation in advisory mode | Safe integration first |
| C2 | C1 + simulator consistency penalty active | Verify dual-consistency effect |
| C3 | C2 + missing-modality stress (camera/gps/power drop) | Robustness in loop |

## 3) Reproducibility Rules

1. Fixed seeds `{2026, 2027, 2028}`.
2. Same time budget per compared method.
3. Keep one config per run and write all outputs to a unique run directory.
4. Report mean/std and paired gains vs baseline.
