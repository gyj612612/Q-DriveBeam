# CCF-A Oriented Research Plan (Public Summary)

## Current Direction

The current public model direction combines:

- object-query-based scene representation
- explicit communication-state encoders
- uncertainty-aware multimodal fusion
- consistency regularization across fused and branch predictions

## Public Novelty Framing

The repository is meant to highlight:

- query-conditioned beam prediction
- robust multimodal fusion under missing modalities
- reproducible ablation and robustness workflows
- an engineering path toward simulator integration

## Experimental Structure

- deterministic configuration-driven runs
- ablation studies for poolers, dropout, and model variants
- robustness checks under sensor degradation and missing-modality stress
- compatibility with downstream CARLA evaluation

## Engineering Rules

- one config controls one run
- outputs are written under `outputs/<exp_name>/`
- modules remain separated by concern: `data`, `models`, `losses`, `train`
- machine-specific datasets and local experiment artifacts stay outside the public commit history

## Public Release Note

Some comparative-baseline details are intentionally kept abstract in the public repository while related paper work is still in progress.
