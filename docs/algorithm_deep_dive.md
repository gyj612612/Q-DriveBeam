# Three-Algorithm Deep Dive (DETR / MM-MIMO-VI / IEMF Family)

## 1) DETR (`facebookresearch/detr`)

Core mechanism:
- CNN backbone extracts image features.
- Transformer encoder-decoder runs on feature map + positional encoding.
- Fixed set of object queries predicts object set directly.
- Hungarian matching aligns predictions with GT one-to-one.
- Loss = class CE + box L1 + GIoU (+ auxiliary decoder-layer losses).

Why it matters for us:
- Query tokens are compact scene structure priors.
- We can transfer pretrained DETR without full COCO retraining.
- Better than global pooled RGB feature for beam decision conditioning.

## 2) MM-MIMO-VI (`ZijianZheng1999/MM-MIMO-VI`)

Repository-level status:
- Minimal prototype style, very few commits.
- Contains VAE modules for camera/LiDAR and VRNN idea for sequence modeling.

Current code risks found locally:
- `camera_vae.py` and `lidar_vae.py` import `model.vae.*` but folder missing.
- `vrnn.py` forward path references `self.rnn_cell` (undefined in class init).
- Indicates concept is useful (generative uncertainty), but repo is not plug-and-play production.

What to absorb (not copy):
- Variational latent modeling for uncertainty-aware representation.
- Sequence-aware latent dynamics for temporal beam prediction.

## 3) IEMF + Versions (local historical line)

Observed progression:
- `version1/version2`: dual-branch baseline + inverse modulation, lower scenario36 performance.
- `version25`: modality gate + richer multi-branch fusion.
- `version3_1_ae` and `version35`: add AE/VAE options, broader encoder/fusion stack.

Empirical pattern from logs:
- Early stage around ~0.31 test accuracy.
- Improved multi-branch runs around ~0.54 to ~0.56.
- Plateau emerges; gain from pure fusion tweaks is limited.

## 4) Unified Direction

Use each line for what it is best at:
- DETR: scene-query representation.
- MM-MIMO-VI idea: uncertainty-aware latent modeling.
- IEMF family: dynamic modality balancing + consistency regularization.

Unified model objective:
- avoid patchwork scripts,
- train as one coherent end-to-end predictor,
- keep direct path to CARLA demo integration.
