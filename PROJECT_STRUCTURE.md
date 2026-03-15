# Q-DriveBeam Structure

## Included in Public Repository

- `README.md`
- `PROJECT_STRUCTURE.md`
- `requirements.txt`
- `.gitignore`
- `configs/`
- `docs/`
- `scripts/`
- `src/beamfusion/`

## Excluded from Public Repository

- `outputs/`
- model checkpoints such as `*.pt` and `*.pth`
- caches such as `__pycache__/` and `.pytest_cache/`
- local datasets and any machine-specific paths

## Core Areas

### `src/beamfusion/`

- configuration loading
- training entry points
- multimodal losses
- utilities
- CARLA-facing adapter
- dataset adapter for `scenario36`
- DETR token extraction and fusion modules

### `scripts/`

- single-run training
- ablation runners
- DETR variant comparison
- top-1 expert baseline
- monitoring and experiment inspection

### `configs/`

- smoke configs
- GPU presets
- ablation and pipeline presets

### `docs/`

- algorithm notes
- experiment matrices
- result summaries
- technical route documents

## Publishing Intent

This public repository is a compact research-engineering snapshot intended to show:

- reproducible experiment structure
- clean multimodal model implementation
- practical training tooling
- connection between perception/fusion research and autonomous-driving simulation
