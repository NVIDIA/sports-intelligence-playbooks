SFT
===

Supervised fine-tuning (SFT) updates the language model weights and multimodal projection layers (vision and audio encoders often stay frozen). Use SFT when you need maximum adaptation capacity (~90–96% of parameters trainable vs. ~0.04–0.17% for LoRA; see :doc:`lora`) and can afford the higher compute and memory of full-parameter training and larger checkpoints (hundreds of GB per save vs. tens of MB for LoRA). Reference GPU memory and throughput numbers are in :ref:`sft-lora-compute-ref` on the :doc:`lora` page.

Both stacks share the same high-level recipe layout under ``configs/``. Cluster settings (Slurm, container, cache directory) live in ``launch_local.yaml``; training hyperparameters live in the recipe YAML.

**Runbooks (commands and smoke examples):**

- **AutoModel:** `automodel/sft/SFT_GUIDE.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/automodel/sft/SFT_GUIDE.MD>`_
- **Megatron-Bridge:** `megatron-bridge/sft/SFT_GUIDE.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/megatron-bridge/sft/SFT_GUIDE.MD>`_

Megatron-Bridge also requires a Megatron-format base checkpoint: `HF→Megatron conversion <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD>`_.

What Gets Trained
-----------------

SFT often keeps vision and audio encoders frozen and trains the LLM plus multimodal projectors (the layers that map vision/sound features into the language model). Use ``freeze_config`` in the recipe YAML (``true`` = frozen, ``false`` = trainable).

.. rubric:: AutoModel

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Parameter
     - Controls
   * - ``freeze_embeddings``
     - Token embedding layer
   * - ``freeze_vision_tower``
     - Vision encoder (RADIO)
   * - ``freeze_audio_tower``
     - Audio / sound encoder
   * - ``freeze_language_model``
     - MoE language model decoder

There are no separate projector freeze flags; vision and audio projectors train when the LM is unfrozen.

.. rubric:: Megatron-Bridge

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Parameter
     - Controls
   * - ``freeze_language_model``
     - MoE language model decoder
   * - ``freeze_vision_model``
     - Vision encoder
   * - ``freeze_vision_projection``
     - Vision-to-LM projector
   * - ``freeze_sound_encoder``
     - Audio / sound encoder
   * - ``freeze_sound_projection``
     - Audio-to-LM projector

Unfreezing encoders increases memory and risk of catastrophic forgetting; keeping encoders frozen is a good starting point.

Training Schedule and Batching
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Parameter
     - **AutoModel** (``step_scheduler``)
     - **Megatron-Bridge** (``step_scheduler``)
   * - Global batch size
     - ``global_batch_size`` — samples per optimizer step across all GPUs
     - ``global_batch_size`` — samples per optimizer step across all GPUs; with ``micro_batch_size`` sets gradient accumulation
   * - Per-GPU microbatch
     - ``local_batch_size`` — how many training examples each GPU processes at once; keep at 1 when sequence packing is on
     - ``micro_batch_size`` — raise only if memory allows; in-batch packing needs ``>= 2``
   * - Max steps
     - ``max_steps`` — hard stop; overrides epoch count
     - ``max_steps`` (``max_iters`` at runtime)
   * - Validation
     - ``val_every_steps``
     - ``val_every_iters``
   * - Checkpoints
     - ``ckpt_every_steps``
     - ``ckpt_every_iters``

**Impact:** Higher global batch size stabilizes MoE training but needs more GPUs for gradient accumulation. More GPUs also let you raise parallelism (TP, EP) to split the model and lower memory per GPU. Fewer ``max_steps`` is fine for smoke tests; production runs typically use hundreds to thousands of steps with cosine decay aligned to ``max_steps``.

Optimizer and Learning Rate
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Parameter
     - **AutoModel**
     - **Megatron-Bridge**
   * - Optimizer
     - ``optimizer``: AdamW, ``lr``, ``weight_decay``, ``betas``
     - ``optimizer``: Adam, ``lr``, ``weight_decay``; optional CPU offload (below)
   * - Learning rate
     - Default 5e-5 for full SFT
     - Default 5e-5 for full SFT
   * - LR schedule
     - ``lr_scheduler``: ``cosine``, ``lr_warmup_steps``, ``lr_decay_steps``, ``min_lr``
     - ``lr_scheduler``: ``cosine``, ``lr_warmup_iters``, ``lr_decay_iters``, ``min_lr``
   * - Grad clip
     - ``clip_grad_norm.max_norm`` (e.g. 1.0)
     - ``clip_grad_norm.max_norm``
   * - Memory savers (Bridge)
     - FSDP ``activation_checkpointing: true``
     - ``optimizer_cpu_offload``, ``optimizer_offload_fraction``, ``use_precision_aware_optimizer``, ``train.empty_unused_memory_level``

.. rubric:: Megatron-Bridge CPU offload

Full SFT can run out of GPU memory even when model weights are sharded across TP/EP. CPU offload moves Adam optimizer state (momentum, variance, and related buffers) from GPU VRAM to host RAM so more room is left for activations and gradients. Tradeoffs: higher host memory use and some overhead when tensors move between GPU and CPU during the optimizer step.

Recipe fields:

- ``optimizer_cpu_offload`` — enable offload (``true`` / ``false``)
- ``optimizer_offload_fraction`` — share of optimizer state on CPU (``1.0`` = all)
- ``use_precision_aware_optimizer`` — must be ``true`` when offload is enabled

Full SFT on a single 8-GPU node often requires enabling CPU offload; multinode runs with more data parallelism may not need CPU offloading depending on the chosen parallelism level. LoRA trains far fewer parameters, so offload is often unnecessary.

Further reading:

- `Megatron Core — Optimizer CPU Offload <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/optimizer_cpu_offload.html>`_ — CPU offload guide
- `Megatron Core OptimizerConfig <https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer_config.html>`_ — offload recipe fields

**Impact:** LR that is too high diverges MoE models quickly; too low underfits. Warmup (often ~10% of total steps) reduces early instability.

Distributed Parallelism
-----------------------

Distributed training has four dimensions of parallelism: **TP**, **PP**, **EP**, and **CP**. For Nemotron Omni, our training scripts support **TP > 1** and **EP > 1** only; keep **PP** and **CP** at ``1``.

Set supported degrees under ``distributed`` in the recipe YAML (or via ``TP`` and ``EP`` env overrides on launch):

- **TP (tensor parallelism)** — splits weight matrices within a layer across GPUs (e.g. attention and MLP shards).
- **EP (expert parallelism)** — shards MoE expert networks across GPUs (required for Nemotron Omni’s MoE decoder).
- **PP (pipeline parallelism)** — splits the model depth-wise across pipeline stages. Not supported for Nemotron Omni in our training scripts; keep ``pp_size`` at ``1``.
- **CP (context parallelism)** — splits sequence length across GPUs for very long contexts. Not supported for Nemotron Omni in our training scripts; keep ``cp_size`` at ``1``.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Parameter
     - AutoModel (``distributed``)
     - Megatron-Bridge (``distributed``)
   * - ``tp_size``
     - Tensor-parallel degree; use ``1`` for Nemotron Omni AutoModel
     - Tensor-parallel degree; ``2`` is typical on 8 GPUs
   * - ``ep_size``
     - Expert-parallel degree; ``8`` on one 8-GPU node (uses DeepEP)
     - Expert-parallel degree; ``8`` (1 node) or ``32`` (multinode)
   * - ``pp_size``
     - Not supported for Nemotron Omni in our training scripts; keep at ``1``
     - Not supported for Nemotron Omni in our training scripts; keep at ``1``
   * - ``cp_size``
     - Not supported for Nemotron Omni in our training scripts; keep at ``1``
     - Not supported for Nemotron Omni in our training scripts; keep at ``1``
   * - Activation checkpoint
     - ``activation_checkpointing: true`` (FSDP)
     - Recompute flags in recipe / env

**Impact:** Let ``world_size = nodes × GPUs_per_node``.

**AutoModel** (Nemotron Omni recipes and training scripts use ``pp_size=1`` and ``cp_size=1``):

- ``world_size`` must be divisible by ``tp_size × pp_size × cp_size``.
- ``ep_size`` must divide ``world_size / pp_size``.
- Data-parallel degree: ``dp_size = world_size / (tp_size × pp_size × cp_size)``.

Example: 1 node × 8 GPUs, ``tp_size=1``, ``pp_size=1``, ``cp_size=1``, ``ep_size=8`` → ``dp_size=8``, and ``8 / 1`` is divisible by ``8``.

**Megatron-Bridge** (Nemotron Omni recipes and training scripts use ``pp_size=1`` and ``cp_size=1``):

- Attention-path data-parallel degree follows the standard Megatron rule: ``dp_size = world_size / (tp_size × pp_size × cp_size)``.
- As a minimum-GPU bound, ``world_size`` must be at least ``pp_size × max(tp_size × cp_size, ep_size)``.

Example: 1 node × 8 GPUs, ``tp_size=2``, ``pp_size=1``, ``cp_size=1``, ``ep_size=8`` → minimum ``= 1 × max(2, 8) = 8``, which matches ``world_size=8``.

Further reading:

- `NeMo AutoModel — Distributed Setup <https://docs.nvidia.com/nemo/automodel/development/distributed-setup>`_ — FSDP2 + expert parallelism
- `Megatron-Bridge — Parallelisms <https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html>`_ — ``tp_size``, ``pp_size``, ``cp_size``, ``ep_size``, ``dp_size``
- `Megatron Core — Parallelism Strategies <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html>`_ — TP, PP, CP, EP

Changing parallelism after training changes the checkpoint layout, so keep ``MODEL_NAME`` and topology consistent to resume.

Data, Video, and Sequence Budget
--------------------------------

Training data is **JSONL** (one JSON object per line) in **HuggingFace Conversation** format: a ``conversation`` array of ``user`` / ``assistant`` turns with typed content (``video``, ``text``, and optionally ``audio``). Rows may also include ``id`` and ``class`` metadata; ``class`` is the question-type / template name (for example ``server_winner``) and is used by eval tooling but is not required for training.

Video (and audio) paths in the JSON are **relative** to ``video_root`` in the recipe YAML. Training resolves each file as ``video_root`` + the relative ``path`` (normalized, no leading slash).

Example line:

.. code-block:: json

   {
     "id": "sample-001",
     "class": "server_winner",
     "conversation": [
       {"role": "user", "content": [
         {"type": "video", "path": "match_id/segment.mp4"},
         {"type": "text", "text": "Who won this point?"}
       ]},
       {"role": "assistant", "content": [
         {"type": "text", "text": "Player A won."}
       ]}
     ]
   }

The ``class`` field is optional for training but recommended in curated datasets so eval can break down accuracy by question type.

The following table summarizes recipe fields that control data loading, video sampling, and sequence budget.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Parameter
     - AutoModel
     - Megatron-Bridge
   * - Train / val JSONL
     - ``dataset`` / ``validation_dataset``: ``path_or_dataset``, ``video_root``
     - ``dataset`` / ``validation_dataset``: ``path_or_dataset``, ``video_root``, ``jsonl_format``
   * - Subsample
     - ``dataset.sample_ratio`` / ``validation_dataset.sample_ratio`` — fraction of JSONL for quick runs
     - ``dataset.sample_ratio`` / ``validation_dataset.sample_ratio`` — fraction of JSONL for quick runs
   * - Video frames
     - ``dataset``: ``max_video_frames``, ``video_sample_fps``, ``min_video_frames`` — one of the main OOM levers
     - ``dataset``: ``max_video_frames``, ``video_sample_fps`` — one of the main OOM levers
   * - Sequence length
     - ``dataloader.packed_sequence.pack_size`` / ``max_length`` (e.g. 24576)
     - ``dataset.seq_length`` (``PACK_SIZE`` env override)
   * - Sequence packing
     - ``dataloader.use_sequence_packing`` + ``packed_sequence.*``
     - ``dataset.pack_sequences_in_batch`` — needs ``micro_batch_size >= 2``

**Sequence packing** improves throughput by fitting multiple short samples in one sequence; packed token count must exceed your longest sample (video + audio + text). See `NeMo AutoModel — Datasets overview <https://docs.nvidia.com/nemo/automodel/datasets/overview>`_ (packed sequence support) and `Megatron-Bridge — Packed sequences <https://docs.nvidia.com/nemo/megatron-bridge/latest/training/packed-sequences.html>`_.

MoE Backend (AutoModel Only)
----------------------------

``model.backend.dispatcher: deepep`` routes expert tokens across GPUs during forward/backward. Pre-Hopper GPUs (e.g. A100) need the matching DeepEP wheel installed at launch (see :doc:`setup` Notes). Using deepep on pre-Hopper, ``ep_size`` is limited to the number of GPUs per node (=8). Override with ``DISPATCHER=torch`` only for debugging (slower).

Weights & Biases
----------------

Training metrics are logged via the ``wandb`` section in the recipe YAML. Set ``entity`` to your W&B account.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Effect
   * - ``mode``
     - ``online`` (needs ``WANDB_API_KEY``), ``offline`` (local logs), or ``disabled``
   * - ``entity`` / ``project`` / ``name``
     - ``entity`` is your W&B account; ``project`` groups related runs; ``name`` is the display label for this run
   * - Bridge fallback
     - Scripts fall back to ``offline`` if ``WANDB_API_KEY`` is unset when mode is ``online``

Quick Start
-----------

1. Complete :doc:`setup` (container, ``launch_local.yaml``, recipe YAML with your data paths).
2. Edit recipe YAML: data paths, ``max_steps``, freeze flags, and memory knobs (frames, ``seq_length`` / ``pack_size``).
3. **AutoModel:** ``launch_interactive_session.sh`` → ``train_interactive.sh``; then ``sbatch_starter.sh`` for scale.
4. **Megatron-Bridge:** HF→Meg conversion once, then the same interactive → sbatch flow.

Smoke commands and CLI override tables are in the code-repo guides above.
