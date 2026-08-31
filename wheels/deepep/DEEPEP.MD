# DeepEP wheels on lustre

DeepEP is a GPU communication library for Mixture-of-Experts (MoE) models. It
provides the token dispatch and combine operations that route activations to
experts across GPUs, reducing communication overhead during MoE training and
inference. In these recipes, `model.backend.dispatcher: deepep` selects DeepEP
as the MoE dispatcher, and its CUDA kernels must be compiled for the target
GPU's streaming-multiprocessor (SM) architecture.

NGC `nemo-automodel:26.06` ships DeepEP for **post-Hopper** GPUs only. On **A100** (SM 8.x),
training with `model.backend.dispatcher: deepep` can fail on the first MoE forward:

```text
RuntimeError: ... layout.cu:128 'named symbol not found'
```

This directory holds **portable `deep_ep` wheels** on lustre plus install/verify scripts.
DeepEP at commit `7febc6e` needs **two wheels** (pre- vs post-Hopper):

| Profile | SM | Typical GPUs | Wheel tag in filename |
|---------|-----|--------------|----------------------|
| `pre-hopper-sm80-sm89` | 8.0, 8.6, 8.9 | A100, A10, L40 | `+7febc6e.prehopper8089` |
| `post-hopper-sm90-sm100-sm120` | 9.0, 10.0, 12.0 | H100, H800, Blackwell | `+7febc6e.posthopper90120` |

## Install

Run **inside** `nemo-automodel:26.06` from repo root. Re-run after each new interactive session.

```bash
cd /path/to/sports_intelligence   # repo root

# Auto-detect from GPU SM (8.x → pre, 9+ → post)
bash wheels/deepep/install_deepep_wheel.sh

# Explicit profile
DEEPEP_WHEEL_PROFILE=pre-hopper-sm80-sm89 bash wheels/deepep/install_deepep_wheel.sh
DEEPEP_WHEEL_PROFILE=post-hopper-sm90-sm100-sm120 bash wheels/deepep/install_deepep_wheel.sh
```

Verify: `/opt/venv/bin/python3 wheels/deepep/check_deepep_wheel.py` (expect `deep_ep OK`).

**Training launch:** SFT/LoRA interactive and batch scripts call `install_deepep_wheel.sh`
automatically when YAML (or `DISPATCHER`) is `deepep`. Skip automatic installation with
`SKIP_DEEPEP_INSTALL=1`; the older `SKIP_DEEPEP_INSTALL_ON_LAUNCH` and
`SKIP_DEEPEP_INSTALL_ON_BATCH` aliases remain supported. Force without a DeepEP recipe
with `DEEPEP_FORCE_INSTALL=1`. Skip the training preflight check with
`SKIP_DEEPEP_WHEEL_CHECK=1`.

## Upstream and license

- **Source:** [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) (branch `hybrid-ep`, commit `7febc6e` in wheel tags)
- **License:** MIT — full copyright and license text is in [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES); each wheel also bundles the upstream LICENSE in `dist-info/licenses/LICENSE`.
- **NVIDIA scripts** in this directory (`install_deepep_wheel.sh`, `check_deepep_wheel.py`, `building/`) are Apache-2.0.
