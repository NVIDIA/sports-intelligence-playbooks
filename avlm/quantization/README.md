# Quantization

Post-training quantization (PTQ) compresses a finished checkpoint so inference uses less GPU memory. The original weights often need a large GPU just to load. Quantized weights are smaller, so the same checkpoint can run on a local GPU. Memory that is no longer spent on weights can hold a larger batch, which raises throughput when a server handles many requests at once.

Recipes live under [`ptq/`](ptq/), with one directory per model. Each directory has its own runbook for the quantization tool, the export format, and how to serve the checkpoint.

Supported models:

- [Nemotron-3-Nano-Omni](ptq/nemotron_3_nano_omni/)

Run commands from the repo root.
