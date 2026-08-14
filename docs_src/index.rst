=================================
Sports Intelligence Cookbooks
=================================

These cookbooks document how to build, train, evaluate, and run **multimodal language models** for sports—models that take video, audio, and text together and answer questions about what happened in a game.

What the Cookbooks Provide
==========================

NVIDIA's public recipes provide generic paths for fine-tuning Nemotron Omni, but they are not tested and verified specifically for analyzing sports feeds. These cookbooks provide a reproducible multimodal language model baseline for sports intelligence, with sports-specific data, evaluation, and distributed operational training and inference workflows already integrated.

1. **Complete sports intelligence starter kit.** Covers the full lifecycle—from sports-video annotation and data preparation through SFT/LoRA training, inference (with quantization and deployment options such as vLLM and SGLang), and evaluation. Its template-driven data pipeline systematically generates multiple training-example types, enabling models to be trained and evaluated across different aspects of sports understanding.

2. **Video- and audio-first fine-tuning with sports-relevant customization.** Provides tested recipes for sports video clips with audio, rather than relying primarily on simpler image-and-text examples available in public recipes. It also exposes input-resolution and frame-sampling settings that help preserve details important to sports analysis, such as player motion, ball trajectory, court geometry, and event timing.

3. **Domain-specific, measurable evaluation.** Includes per-question-class MCQ metrics and LLM-as-a-judge evaluation for captions and descriptive answers.

4. **Reproducible, scalable execution.** Supplies tested distributed workflows, checkpoint conversion and parity checks, and automated inference-to-evaluation pipelines.

5. **Informed choice of training method and framework.** Supports and compares full SFT and LoRA across NeMo AutoModel and Megatron-Bridge, helping users balance quality, memory, iteration speed, and checkpoint size. When properly tuned, Megatron-Bridge can deliver up to **3x higher training throughput** than AutoModel.

6. **Last but not least: shared practical learnings.** Shares lessons from building, debugging, optimizing, and validating multimodal training workflows that can help others fine-tune multimodal language models for sports intelligence.

Implementation, launch scripts, and runbooks live in the `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_ code repo under ``avlm/training/``, ``avlm/inference/``, and ``avlm/evals/``.

Use the navigation on the left to browse by section.

.. toctree::
   :caption: Multimodal Language Models
   :maxdepth: 3

   multimodal/data_curation
   multimodal/training/index
   multimodal/inference/index
   multimodal/evaluation
   multimodal/learnings

.. toctree::
   :caption: Reference
   :hidden:

   licenses
