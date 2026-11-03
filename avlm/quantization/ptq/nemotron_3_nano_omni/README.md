# Nemotron-3-Nano-Omni PTQ

Post-training quantization for **Nemotron-3-Nano-Omni** BF16 checkpoints: the public reasoning model, or a consolidated supervised-fine-tuning export such as a sports checkpoint.

The recipes in [`modelopt/`](modelopt/) use [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer). That repository is a library of methods for compressing a trained model, including quantization, distillation, pruning, neural architecture search, and speculative decoding. The compressed model is meant for inference frameworks such as vLLM and TensorRT. These recipes use the quantization path: calibrate a finished Hugging Face checkpoint and export **FP8** or **NVFP4** weights. vLLM serves FP8 with `quantization=modelopt` and NVFP4 with `modelopt_fp4`.

Runbook: [`modelopt/README.md`](modelopt/README.md).
