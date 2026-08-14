Training
========

Sports intelligence depends on models that can reason in all of the visual, audio/speed, and text modalities. Fine-tuning multimodal language models on domain-specific data is one of the most effective ways to improve accuracy: sports analysis relies on fine-grained details such as player motion, ball trajectory, court geometry, and timing that general-purpose models often miss. Adequate spatial and temporal resolution in both data and model inputs is essential so the model can ground answers in what actually happened in the clip. Getting the most accurate results therefore requires more than tuning the weights of a fixed base model alone. It also depends on how multimodal data is curated, encoded, and aligned across the full fine-tuning pipeline.

This training section covers supervised fine-tuning (SFT) and LoRA (Low-Rank Adaptation) for `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 <https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16>`_. Our recommended "Nemotron-3 Nano Omni" model is available for commercial use. Our cookbooks and launch scripts in the code repo are tested against this base model.

See :doc:`setup` for stack layout and prerequisites. SFT and LoRA link to full runbooks are in the code repo.

.. toctree::
   :maxdepth: 1

   setup
   sft
   lora
