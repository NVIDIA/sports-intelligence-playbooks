LoRA
====

LoRA (Low-Rank Adaptation) trains small adapter matrices injected into selected layers instead of updating all language-model weights. Use LoRA for faster iteration, smaller checkpoints, and lower optimizer memory while keeping the base model frozen.

With our default recipes on Nemotron-3 Nano Omni (~33B total parameters), LoRA updates well under 1% of model weights. Checkpoints store adapter weights only, typically tens of MB per save, while full SFT on the same topology writes hundreds of GB per save. See :ref:`sft-lora-compute-ref` for trainable-parameter, memory, and throughput comparisons.

**Inference:** LoRA checkpoints are not standalone models. You always need the frozen base checkpoint plus the adapter (HF ``peft`` load for AutoModel; Megatron base + LoRA ``iter_*`` for Bridge). On Megatron-Bridge you can alternatively merge adapters into the base with ``merge_lora.py`` to produce a single deployable checkpoint (see :ref:`lora-post-training`).

.. _sft-lora-compute-ref:

SFT vs. LoRA (compute and checkpoint size)
------------------------------------------

The table below is an approximate SFT vs. LoRA comparison for Nemotron-3 Nano Omni. Actual numbers depend on your recipe, sequence length, and hardware.

GPU memory and throughput (tps/gpu) come from the `Nemotron-Omni AutoModel cookbook <https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/nemotron-omni>`_ (8 GPUs, CORD-V2 recipe, frozen vision/audio towers).

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Metric
     - SFT
     - LoRA
   * - GPU memory per device (steady-state)
     - ~49 GiB (~1.6× LoRA)
     - ~30 GiB
   * - Throughput (tps/gpu)
     - ~2.4k–2.6k
     - ~2.5k–3.3k (comparable)
   * - Checkpoint per ``iter_*`` save
     - hundreds of GB
     - tens of MB (e.g. ~26 MB)
   * - Trainable parameters
     - ~90–96% (LLM + projectors; encoders frozen)
     - ~0.04% at rank 16 (~14M) / ~0.17% at rank 64 (~55M)

Trainable LoRA parameters scale linearly with ``peft.dim``; AutoModel and Megatron-Bridge train the same count at a given rank. Percentages are relative to ~33B total parameters; rank 64 figures come from the Nemotron-Omni cookbook.

Step time is often similar between SFT and LoRA because throughput (tps/gpu) is in the same ballpark; the main SFT costs are higher per-GPU memory, optimizer state over far more trainable weights, and checkpoint I/O.

Both stacks share the same high-level recipe layout under ``configs/``. Cluster settings (Slurm, container, cache directory) live in ``launch_local.yaml``; training hyperparameters live in the recipe YAML.

**Runbooks (commands, merge/export, smoke examples):**

- **AutoModel:** `automodel/lora/LORA_GUIDE.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/automodel/lora/LORA_GUIDE.MD>`_
- **Megatron-Bridge:** `megatron-bridge/lora/LORA_GUIDE.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/megatron-bridge/lora/LORA_GUIDE.MD>`_

Megatron-Bridge also requires a Megatron-format base checkpoint: `HF→Megatron conversion <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD>`_.

Choose SFT or LoRA based on how much of the model you need to adapt and your operational constraints; see :ref:`sft-lora-compute-ref` for memory, throughput, and checkpoint tradeoffs.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Aspect
     - SFT
     - LoRA
   * - Adaptation scope
     - Full LLM weights + multimodal projectors
     - Low-rank adapters on LM linear layers only
   * - Best for
     - Maximum quality, production models
     - Rapid experiments, resource-constrained runs

What Gets Trained (Target Modules and Freeze)
---------------------------------------------

Encoders often stay frozen; adapters attach to **language-model** linear layers.

.. rubric:: AutoModel

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Role
   * - ``dim``
     - LoRA **rank**: inner dimension of the low-rank adapter (update approximated as ``ΔW = B·A``). Higher rank means more adapter capacity and trainable parameters. Default **64** in AutoModel recipes.
   * - ``alpha``
     - LoRA **scaling**: scales how strongly the adapter is applied (effective scale ≈ ``alpha / dim``). Default **128** in AutoModel recipes (``alpha / dim = 2``).
   * - ``match_all_linear: false``
     - Do not attach to every linear layer
   * - ``exclude_modules``
     - Glob patterns skipping vision, audio, sound, ``lm_head``, projector (``*mlp1*``)

**Freeze:** same encoder freezes as SFT; base LM weights stay fixed while adapters train (``freeze_language_model: false`` allows adapter gradients through the LM stack).

.. rubric:: Megatron-Bridge

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Role
   * - ``dim`` / ``alpha``
     - Same meaning as AutoModel (under ``peft``). Defaults **16** / **32**.
   * - ``linear_qkv``
     - Attention Q/K/V projections
   * - ``linear_proj``
     - Attention output projection
   * - ``in_proj`` / ``out_proj``
     - Mamba / SSM-style blocks in the decoder (under ``peft.target_modules``)

Freeze defaults for LoRA: encoders and vision/sound projections stay frozen (``freeze_vision_projection: true``, ``freeze_sound_projection: true``); only the LoRA adapters update the language model.

**Impact:** Broader target sets (higher rank, more modules) increase capacity and memory. Narrow targets train faster but may underfit complex sports QA. Excluding vision/audio modules prevents wasting capacity on frozen towers.

Training Schedule and Batching
------------------------------

Same step scheduler fields as :doc:`sft` (global batch size, per-GPU microbatch, max steps, validation, checkpoints). LoRA can often use a larger global batch size compared to SFT because the adapter state is smaller.

Optimizer and Learning Rate
---------------------------

LoRA uses a **higher learning rate** than full SFT because only adapter parameters update. Optimizer type, weight decay, LR schedule, and grad clip follow the same recipe fields as :doc:`sft`; defaults differ only in LR:

- **AutoModel:** **1e-3** (vs. 5e-5 for SFT)
- **Megatron-Bridge:** **1e-4** (vs. 5e-5 for SFT)

**Impact:** If loss spikes, reduce LR or increase warmup. LoRA often tolerates 10–20× the SFT learning rate because the update space is low-rank. CPU optimizer offload (see :doc:`sft`) is usually unnecessary for LoRA due to lower GPU memory usage.

Distributed Parallelism
-------------------------

Same rules as :doc:`sft` (``tp_size``, ``ep_size``, ``pp_size``, ``cp_size``, and ``world_size`` constraints).

Data, Video, and Sequence Budget
--------------------------------

Same JSONL format, ``video_root`` resolution, dataset fields, and sequence-packing rules as :doc:`sft`. LoRA recipes reuse those fields unchanged.

MoE Backend (AutoModel Only)
----------------------------

``model.backend.dispatcher: deepep`` routes expert tokens across GPUs during forward/backward. Pre-Hopper GPUs (e.g. A100) need the matching DeepEP wheel installed at launch (see :doc:`setup` Notes). Using deepep on pre-Hopper, ``ep_size`` is limited to the number of GPUs per node (=8). Override with ``DISPATCHER=torch`` only for debugging (slower).

Checkpointing
-------------

Training writes adapter state only, not the full base model:

- **AutoModel:** ``adapter_model.safetensors`` plus small metadata under ``lora/slurm/outputs/<MODEL_NAME>/`` (typically tens of MB per step save); ``save_consolidated: true`` for HF-friendly layout
- **Megatron-Bridge:** ``save_optim: false`` by default; ``iter_*`` Megatron ``.distcp`` shards holding LoRA weights only (tens of MB total per ``iter_*``, vs. hundreds of GB for SFT on the same ``tp``/``ep`` layout)

The frozen base always comes from ``model.hf_model_id`` (AutoModel) or ``checkpoint.pretrained_checkpoint`` (Bridge) at train and inference time. Adapter-only checkpoints cannot be loaded alone — pair them with the base, or merge (Bridge) before deployment.

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

.. _lora-post-training:

Post-Training (Megatron-Bridge Only)
------------------------------------

After LoRA training, merge or export adapters with Megatron-Bridge example scripts included in the NeMo container. By default they are under ``${MEGATRON_BRIDGE_ROOT}`` (``/opt/Megatron-Bridge``):

1. **Merge** adapters into base Megatron weights — ``${MEGATRON_BRIDGE_ROOT}/examples/peft/merge_lora.py`` (`merge_lora.py <https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_omni/examples/peft/merge_lora.py>`_)
2. **Export** HF PEFT adapter — ``${MEGATRON_BRIDGE_ROOT}/examples/conversion/adapter/export_adapter.py`` (`export_adapter.py <https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_omni/examples/conversion/adapter/export_adapter.py>`_)

If you cloned Megatron-Bridge to Lustre (``MEGATRON_BRIDGE_GIT_BOOTSTRAP=1``), use the same paths under your clone root.

Quick Start
-----------

1. Complete :doc:`setup` (container, ``launch_local.yaml``, recipe YAML with your data paths).
2. Pick a recipe under ``lora/configs/`` via ``CONFIG_YAML_REL`` or ``CONFIG_YAML=``; tune ``peft`` rank/targets and ``optimizer.lr`` if adapting a new task.
3. **AutoModel:** ``launch_interactive_session.sh`` → ``train_interactive.sh``; then ``sbatch_starter.sh`` for scale.
4. **Megatron-Bridge:** HF→Meg conversion once, then the same interactive → sbatch flow; use ``RESUME_CHECKPOINT=1`` to continue.

Smoke commands and CLI override tables are in the code-repo guides above.
