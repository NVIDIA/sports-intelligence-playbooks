# Post-training quantization (PTQ)

PTQ quantizes a checkpoint after training is finished. The weights are not updated. A calibration pass runs the model forward on a small set of samples, measures the range of those values, and stores the resulting scales in the exported checkpoint.

Quantization-aware training (QAT) happens while the model is still training or being fine-tuned. The training loop simulates low precision so the weights can adapt to it, which takes a training dataset and a full training run. 

The recipes here are static PTQ: calibration chooses the scales before the checkpoint is served.

Supported models:

- [Nemotron-3-Nano-Omni](nemotron_3_nano_omni/)
