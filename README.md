# CoPD: Co-Evolving Policy Distillation

Implementation of CoPD with two- and three-branch training. Each branch learns
from its own data with GRPO and exchanges knowledge through on-policy distillation.
Three-branch training uses a text hub with image and video spokes;
communication is limited to hub–spoke pairs.

## Contents

- `verl/trainer/`: two- and three-branch CoPD trainers.
- `configs/copd.yaml`: 4B training configuration.
- `scripts/train_dual.sh`: two-branch training launcher.
- `scripts/model_merger.py` and `scripts/weight_interpolate.py`: checkpoint export and model fusion.
- `prompts/` and `rewards/`: training prompt and reward functions.

## Installation

Use Python 3.11 with CUDA 12.8. FlashAttention is required and installed last.

```bash
uv venv --python 3.11.14
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Training

Set the model, data paths, `BK` and GPU allocation in
[scripts/train_dual.sh](scripts/train_dual.sh), then run:

```bash
bash scripts/train_dual.sh
```

Other training settings are in [configs/copd.yaml](configs/copd.yaml).

## Citation

## Acknowledgement

This repo builds on [EasyVideoR1](https://github.com/cyuQ1n/EasyVideoR1),
[EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl).
Thanks to their great work!

Licensed under [Apache-2.0](LICENSE). See [NOTICE.md](NOTICE.md) for source attribution.
