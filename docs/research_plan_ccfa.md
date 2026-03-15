# CCF-A Oriented Research Plan (2026)

## 1) Current Baseline Reality (from local logs)

Observed scenario36 test accuracy bands:
- early versions (`version1/version2/...`): around `0.31`.
- improved multi-branch variants: around `0.54~0.56`.
- strongest existing run snapshot: about `0.5569`.

Conclusion:
- Existing IEMF-style gates improved substantially.
- Performance now appears to be in a local plateau.
- Next gain needs stronger scene understanding and robustness, not another shallow fusion tweak.

## 2) Unified Method (not copy-paste of DETR/MM-MIMO-VI)

Proposed model: `Q-DriveBeam`
- `DETR scene query tokens` from camera views (`rgb5/rgb6`).
- `GPS encoder` + `Power encoder` for communication state.
- `Uncertainty-gated fusion` (learned prior + inverse-uncertainty weighting).
- `Consistency regularization` between fused prediction and branch predictions.
- Optional simulator-physics consistency term in next stage.

Core principle:
- Use DETR as a pretrained scene prior.
- Keep communication branches explicit.
- Learn fusion under uncertainty and enforce agreement.

## 3) CCF-A Level Novelty Candidates

Candidate novelty package for submission:
- Query-conditioned beam prediction:
  beam distribution predicted from object-set tokens rather than single pooled image feature.
- Dual-consistency training:
  model consistency (fused vs branch) + simulator consistency (predicted beam vs channel/trajectory constraints).
- Robust missing-modality regime:
  train-time modality dropout + uncertainty-aware gate reweighting.
- Transfer to autonomous-driving demo:
  same architecture running on CARLA stream with online beam recommendation.

## 4) Experimental Protocol

Mandatory:
- Deterministic splits (fixed seeds, reported mean/std over 3-5 seeds).
- Same train budget for all baselines.
- Ablation on:
  1) no DETR token branch,
  2) no uncertainty gate,
  3) no consistency,
  4) single-view vs dual-view.
- Robustness:
  1) camera blur/occlusion,
  2) GPS noise,
  3) power signal corruption.

## 5) Engineering Rules

- One config controls one run.
- All outputs saved to `outputs/<exp_name>/`.
- No hidden hard-coded paths except config defaults.
- Keep modules separated: `data`, `encoders`, `fusion`, `losses`, `train`.
