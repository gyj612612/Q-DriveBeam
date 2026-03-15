# Paper Results Table v1 (Initial Draft)

Date: 2026-02-26

## 1) Offline Scenario36 Results (existing local baselines)

| Method / Source | Test Acc (Top-1) | Notes |
|---|---:|---|
| IEMF v1 (Normal) | 0.3407 | `C:\brain-inspired mechanisms\analysis\6g_logs_multibranch\version1_original\scenario36_results.json` |
| IEMF v1 (IEMF) | 0.3370 | Same run file |
| Earlier dual-branch baseline (Normal) | 0.5392 | `C:\brain-inspired mechanisms\analysis\6g_logs\scenario36_results.json` |
| Earlier dual-branch baseline (IEMF) | 0.5410 | Same run file |
| Best local multi-branch (v25 full, Normal) | 0.5529 | `...version25_latest_full/scenario36_multibranch_results.json` |
| Best local multi-branch (v25 full, IEMF) | **0.5569** | Current strongest offline local result |

## 2) Closed-loop CARLA Result (newly rerun)

Run ID:
- `E:\6G\carla_data\run_20260226_042637`

Command family:
- Traditional autonomous-driving baseline (Town05, ego-adv, no-save).

| Metric | Value |
|---|---:|
| Collision counter (final) | **0** |
| Avg speed (km/h) | 26.60 |
| Max speed (km/h) | 47.86 |
| Approx FPS | 8.55 |
| Lane-change ratio | 0.00 |
| Logged sample rows | 23 |

## 3) Reading of Current Status

1. Existing offline beam prediction already reaches ~0.557 top-1 locally.
2. Traditional CARLA controller can run collision-free, but still needs speed-progress robustness under harder stress.
3. Next table version (v2) should add:
   - A0~A5 ablations,
   - R1~R4 robustness,
   - C0~C3 closed-loop comparisons with BeamFusion integration.

## 4) New Quick Ablations (A6/A7 sanity, 2026-02-26)

Note:
- These are quick debug runs (`max_samples=512`, `epochs=2`, short train/val steps).
- They are for direction check only, not final paper numbers.

### A6 (Query pooler)

| Setting | best_val_acc | test_acc | test_loss | Output |
|---|---:|---:|---:|---|
| `score_weighted_mean` | 0.3250 | **0.2625** | **4.7042** | `E:\6G\Code\outputs\ablation_a6_a7\quick_a6_smoke\01_a6_score_weighted_mean` |
| `attn_pool` | 0.3000 | **0.2625** | 5.0958 | `E:\6G\Code\outputs\ablation_a6_a7\quick_a6_smoke\02_a6_attn_pool` |
| `cls_cross_attn` | 0.3000 | 0.2250 | 4.9841 | `E:\6G\Code\outputs\ablation_a6_a7\quick_a6_smoke\03_a6_cls_cross_attn` |

Quick read:
- In this short-budget setting, `score_weighted_mean` is strongest.
- `attn_pool` ties on test_acc but has worse test_loss; should still be kept for full-budget verification.

### A7 (modality dropout, base pool=`score_weighted_mean`)

| `modality_dropout_p` | best_val_acc | test_acc | test_loss | Output |
|---:|---:|---:|---:|---|
| 0.0 | **0.3250** | **0.2625** | **4.7042** | `E:\6G\Code\outputs\ablation_a6_a7\quick_a7_smoke\01_a7_drop_0.0` |
| 0.1 | 0.2875 | 0.2250 | 5.0971 | `E:\6G\Code\outputs\ablation_a6_a7\quick_a7_smoke\02_a7_drop_0.1` |
| 0.2 | 0.3000 | 0.2250 | 4.9532 | `E:\6G\Code\outputs\ablation_a6_a7\quick_a7_smoke\03_a7_drop_0.2` |
| 0.3 | 0.2750 | 0.2000 | 5.2291 | `E:\6G\Code\outputs\ablation_a6_a7\quick_a7_smoke\04_a7_drop_0.3` |

Quick read:
- With limited training budget, stronger modality dropout hurts clean-set accuracy.
- This is expected in early stage; robustness gain must be judged on dedicated missing-modality stress tests (R4), not clean test only.

## 5) New Multi-seed Quick Sanity (A6, seeds=2026/2027)

Source:
- `E:\6G\Code\outputs\ablation_a6_a7\quick_a6_2seed\results.md`

| Query pooler | best_val_acc (mean+/-std) | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---|---:|---:|---:|
| `score_weighted_mean` | **0.2875 +/- 0.0530** | **0.2688 +/- 0.0088** | **4.7460 +/- 0.0591** |
| `attn_pool` | 0.2625 +/- 0.0530 | 0.2500 +/- 0.0177 | 4.9112 +/- 0.2611 |
| `cls_cross_attn` | 0.2562 +/- 0.0619 | 0.2313 +/- 0.0088 | 4.8706 +/- 0.1604 |

Read:
- In this quick multi-seed setting, `score_weighted_mean` is still the most stable first-choice baseline.
- `attn_pool` remains a secondary candidate to retest under full budget.

## 6) New DETR Variant Transfer Sanity (A8, quick, seed=2026)

Source:
- `E:\6G\Code\outputs\detr_variant_compare\quick_a8_variants\results.md`

| Variant | test_acc | test_loss | Read |
|---|---:|---:|---|
| R50 (hub pretrained) | 0.2625 | 4.7042 | Current stable baseline line |
| R101 (local ckpt) | **0.2875** | 4.6108 | Better top-1 than R50 in this quick setting |
| R101-DC5 (local ckpt) | **0.2875** | 4.6102 | Similar to R101, marginally lower loss |
| R101-panoptic (local ckpt) | 0.2750 | **4.5473** | Better loss, but top-1 below R101/R101-DC5 |

Read:
- Quick result suggests R101 / R101-DC5 transfer is promising for our task.
- Final conclusion requires multi-seed full-budget confirmation before paper claim.

## 7) Full-budget A6 Progress (seed=2026 completed)

Source:
- `E:\6G\Code\outputs\ablation_a6_a7\full_a6_s2026\results.md`

| Query pooler | best_val_acc | test_acc | test_loss |
|---|---:|---:|---:|
| `score_weighted_mean` | 0.3583 | 0.3354 | 3.8116 |
| `attn_pool` | **0.3812** | **0.3521** | **3.6718** |
| `cls_cross_attn` | 0.3521 | 0.3271 | 3.7822 |

Read:
- Under full budget (seed=2026), `attn_pool` outperforms other poolers on val/test accuracy and loss.
- This reverses quick-run ranking and confirms that full-budget evaluation is essential.
- Multi-seed full-budget confirmation is running with GPU config (`seed=2027,2028`).

## 8) Full-budget A6 Multi-seed (seeds=2026/2027/2028, completed)

Source:
- `E:\6G\Code\outputs\ablation_a6_a7\ccfa_auto_v1_a6\results.md`

| Query pooler | best_val_acc (mean+/-std) | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---|---:|---:|---:|
| `score_weighted_mean` | **0.3361 +/- 0.0151** | 0.3215 +/- 0.0229 | **3.8210 +/- 0.1099** |
| `attn_pool` | 0.3326 +/- 0.0170 | 0.3181 +/- 0.0151 | 3.8871 +/- 0.1129 |
| `cls_cross_attn` | 0.3333 +/- 0.0021 | **0.3243 +/- 0.0170** | 3.8578 +/- 0.0618 |

Read:
- `cls_cross_attn` gives the highest mean test accuracy in this full-budget multi-seed setting.
- `score_weighted_mean` is strongest on validation mean and mean loss, with slightly lower test mean than `cls_cross_attn`.
- This indicates a close tradeoff and justifies carrying both into robustness checks.

## 9) New Runtime Optimization Validation (2026-03-01)

Source:
- `E:\6G\Code\outputs\ccfa_pipeline\ccfa_fast1h_v1_live_status.json`
- `E:\6G\Code\docs\ccfa_pipeline_report_auto.md`

Observed wall-clock:
- pipeline start: `2026-03-01T06:38:41Z`
- pipeline end: `2026-03-01T06:54:45Z`
- total: about **16 minutes**

Read:
- Fast screening profile now finishes well within 1 hour on RTX 4060.
- This supports a two-stage protocol:
  1) fast screening every day,
  2) paper-budget reruns only for shortlisted candidates.

## 10) New A8 Reproducible Multi-seed Table (uniform budget, 2026-03-01)

Source:
- `E:\6G\Code\outputs\detr_variant_compare\ccfa_a8_repro_20260301\results.md`

| Variant | test_acc (mean+/-std) | test_loss (mean+/-std) | Note |
|---|---:|---:|---|
| R50 (hub pretrained) | 0.3014 +/- 0.0277 | 4.0158 +/- 0.0306 | stronger mean val than R101 in this budget |
| R101 (local ckpt) | **0.3069 +/- 0.0339** | 3.9960 +/- 0.1465 | best mean Top-1 |
| R101-DC5 (local ckpt) | 0.3056 +/- 0.0324 | **3.9919 +/- 0.1300** | best mean loss |

Read:
- Under the current fast paper-budget (with DETR cache + early stop), the three variants are close.
- R101 has slight edge in Top-1 mean; R101-DC5 has slight edge in loss.
- Next action should be fixed-budget reruns for final paper claim (same seeds, no protocol drift).
