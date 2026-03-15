# Literature Map (for Method Positioning)

## A. Vision Query Learning

1. DETR (ECCV 2020)
- Paper: https://arxiv.org/abs/2005.12872
- Key idea: set prediction with bipartite matching; no NMS.

2. Deformable DETR (ICLR 2021)
- Paper: https://arxiv.org/abs/2010.04159
- Key idea: sparse multi-scale attention for faster convergence and better small-object handling.

3. DINO for DETR (ICLR 2023)
- Paper: https://arxiv.org/abs/2203.03605
- Key idea: denoising anchors and stronger training for DETR family.

## B. Autonomous-Driving Multimodal Fusion (CCF-A relevant CV venues)

4. BEVFormer (ECCV 2022)
- Paper: https://arxiv.org/abs/2203.17270
- Key idea: camera-based BEV with spatiotemporal transformer.

5. BEVFusion (NeurIPS 2022)
- Paper: https://arxiv.org/abs/2205.13542
- Key idea: unified LiDAR-camera BEV fusion with strong robustness.

6. UniAD (CVPR 2023)
- Paper: https://arxiv.org/abs/2212.10156
- Key idea: planning-oriented end-to-end autonomous driving stack.

## C. 6G / Beam Prediction Context

7. DeepSense 6G Challenge Track 2
- Main page: https://www.deepsense6g.net/challenge2025/
- Public deep-learning beam prediction benchmark and tracks.

8. DeepSense-public baseline/paper index
- Repo: https://github.com/wigig-tools/Deepsense6G-papers
- Useful for identifying realistic beam-prediction baselines.

9. 2025 challenge paper (multimodal beam prediction baseline example)
- ArXiv entry listed by DeepSense index:
  https://arxiv.org/abs/2507.00126

## D. Positioning for Our Paper

Target gap:
- Existing beam predictors often use pooled visual features.
- Existing DETR papers are not optimized for communication-beam objectives.

Our position:
- Query-level scene tokens + communication modalities + uncertainty/consistency training.
- One model family reusable in both offline dataset experiments and CARLA online demo.
