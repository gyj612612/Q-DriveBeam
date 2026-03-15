# Architecture Positioning

`Q-DriveBeam` is presented as a clean research-engineering codebase centered on three public-facing ideas:

- scene-query visual representation
- uncertainty-aware multimodal fusion
- consistency-regularized training under missing-modality stress

## Public Release Scope

This repository includes the main training, ablation, and deployment-facing code needed to understand the project structure and reproduce the public engineering workflow.

## Private Scope

Some comparative-baseline details and paper-in-progress implementation lineage are intentionally abstracted in the public release.

This keeps the repository:

- suitable for portfolio and application review
- technically representative
- compatible with ongoing manuscript work

## Practical Reading Guide

For public review, the most relevant files are:

- `src/beamfusion/models/model.py`
- `src/beamfusion/models/fusion.py`
- `src/beamfusion/train.py`
- `scripts/run_ablation_a6_a7.py`
- `scripts/run_detr_variant_compare.py`
- `src/beamfusion/carla_adapter.py`
