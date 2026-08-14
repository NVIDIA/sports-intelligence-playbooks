Inference
=========

After fine-tuning, we run inference to generate predictions and evaluate model quality. Keep the video preprocessing and text prompt style/template consistent with training: frame rate (``video_fps``) and resolution settings affect how much spatial and temporal detail the model sees, and mismatches in preprocessing or prompting can shift results on fine-grained sports QA.

This section covers distributed inference (interactive and ``sbatch``) for our recommended base model `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 <https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16>`_ and checkpoints from our training stacks. Use the backend that matches how the checkpoint was produced: **NeMo AutoModel** for HuggingFace weights (base, SFT, or LoRA), or **Megatron-Bridge** for Megatron-format.

Full command examples live in `avlm/inference/README.md <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/inference/README.md>`_.

Setup
-----

Inference code lives in the `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_ repo at ``avlm/inference/`` (shared driver: ``common/run_inference.py``). Clone the repo, ``cd`` to its root, and run all launch commands from there.

Containers
~~~~~~~~~~

Set ``CONTAINER_IMAGE`` in each backend's ``slurm/launch_local.yaml`` (copy from ``launch.yaml``).

- **AutoModel inference:** `nemo-automodel:26.06 <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/26.06>`_
- **Megatron-Bridge inference:** `nemo:26.06 <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/26.06>`_

On Slurm with **enroot**, convert the NGC image once to **``.sqsh``** (see :doc:`../training/setup`). Point ``CONTAINER_IMAGE`` at the resulting path.

Two Backends (Pick One)
~~~~~~~~~~~~~~~~~~~~~~~

Pick one of the following based on the framework you used for training.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - **NeMo AutoModel** (HF)
     - **Megatron-Bridge** (Megatron)
   * - Checkpoint
     - HF base, full SFT weights, or LoRA adapter + base
     - Megatron ``iter_*`` dir from Bridge SFT (or merged LoRA)
   * - Container
     - `nemo-automodel:26.06 <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/26.06>`_
     - `nemo:26.06 <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/26.06>`_
   * - Launcher path
     - ``avlm/inference/automodel/slurm/``
     - ``avlm/inference/megatron-bridge/slurm/``
   * - Example configs
     - ``automodel/configs/default_sft.yaml``, ``default_lora.yaml``
     - ``megatron-bridge/configs/default_mbridge.yaml``

Shared Layout
~~~~~~~~~~~~~

- **common/run_inference.py** — JSONL inference driver (both backends)
- **common/models/** — backend-specific model loaders
- **common/utils/_inference_lib.sh** — CLI/env overrides for launch scripts
- **Per-backend configs/ + slurm/** — launch_local.yaml, interactive + sbatch launchers

**Outputs:** ``avlm/inference/outputs/<INFERENCE_NAME>/``  
**Logs (batch):** ``avlm/inference/logs/run_<timestamp>/``

Interactive vs. Sbatch
~~~~~~~~~~~~~~~~~~~~~~

Start with **interactive inference on 1 node** and a small ``MAX_INFERENCE_SAMPLES`` smoke before larger ``sbatch`` jobs.

Typical Workflow
~~~~~~~~~~~~~~~~

1. Copy ``launch.yaml`` → ``launch_local.yaml`` under the backend's ``slurm/`` dir; set ``CONTAINER_IMAGE``, ``CACHE_DIR``, and Slurm accounts.
2. Edit an inference YAML: eval JSONL ``data_path``, ``video_root``, and checkpoint path.
3. ``launch_interactive_session.sh`` → ``infer_interactive.sh`` with ``INFERENCE_CONFIG=...``.
4. Scale with ``sbatch_starter.sh`` when the smoke passes.

AutoModel Inference
-------------------

HF-format inference via ``common/run_inference.py`` with ``model_name: nemotron_omni_hf``. Supports the HF base model, full SFT checkpoints, and LoRA adapters.

Quick Start
~~~~~~~~~~~

.. code-block:: bash

   bash avlm/inference/automodel/slurm/interactive/launch_interactive_session.sh

   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft.yaml \
   MAX_INFERENCE_SAMPLES=16 \
   bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh

For batch:

.. code-block:: bash

   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft.yaml \
   num_nodes=2 \
   bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh

Override a trained SFT checkpoint:

.. code-block:: bash

   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft.yaml \
   MODEL_PATH=avlm/training/automodel/sft/slurm/outputs/<MODEL_NAME>/checkpoints_.../epoch_0_step_849 \
   bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh

Config Files
~~~~~~~~~~~~

- **Full SFT / base:** ``automodel/configs/default_sft.yaml`` (``is_lora: false``)
- **LoRA:** ``automodel/configs/default_lora.yaml`` (``is_lora: true``)

Megatron-Bridge Inference
-------------------------

Inference over Megatron-format checkpoints from Bridge SFT (or a merged LoRA checkpoint). ``megatron-bridge/entrypoint.py`` applies the Bridge AVLM overlay, then calls ``common/run_inference.py`` with ``model_name: nemotron_omni_mbridge``.

Quick Start
~~~~~~~~~~~

.. code-block:: bash

   bash avlm/inference/megatron-bridge/slurm/interactive/launch_interactive_session.sh

   INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_mbridge.yaml \
   MAX_INFERENCE_SAMPLES=16 \
   bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh

For batch:

.. code-block:: bash

   INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_mbridge.yaml \
   num_nodes=2 \
   bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh

Set ``model_path`` to your training checkpoint directory (for example ``.../iter_0003100``). Match the parallelism fields to how the checkpoint was trained. Optional: ``MEGATRON_BRIDGE_GIT_BOOTSTRAP=1`` on session launch if not using ``/opt/Megatron-Bridge`` from the container (same as training).

Config file: ``megatron-bridge/configs/default_mbridge.yaml``.

Inference Config Reference
--------------------------

Shared Fields (Both Backends)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Role
   * - ``inference_name``
     - Output subdirectory name under ``base_output_dir``; also used as the Slurm job name
   * - ``data_path``
     - Eval JSONL (HF conversation format)
   * - ``video_root``
     - Root directory prepended to relative video paths in each JSONL row
   * - ``base_output_dir``
     - Parent directory for outputs (default ``avlm/inference/outputs/``)
   * - ``max_new_tokens``
     - Maximum tokens generated per sample
   * - ``video_fps``
     - Frame sampling rate when decoding video clips
   * - ``max_video_frames``
     - Maximum frames passed to the vision encoder per clip. Sampling first targets ``video_fps`` (roughly clip duration × fps frames); if that count exceeds this cap, frames are reduced to ``max_video_frames`` by **uniform subsampling** (evenly spaced indices across the clip)

CLI / Environment Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are applied on top of the inference YAML (see `avlm/inference/README.md <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/inference/README.md>`_ for examples).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Override
     - Role
   * - ``INFERENCE_CONFIG``
     - Path to the inference YAML
   * - ``INFERENCE_NAME``
     - Output subdirectory name (overrides ``inference_name`` in the YAML)
   * - ``DATA_PATH``
     - Eval JSONL path
   * - ``MODEL_PATH``
     - Checkpoint path (overrides ``model_path``)
   * - ``ADAPTER_PATH``
     - LoRA adapter directory (AutoModel LoRA only)
   * - ``BASE_OUTPUT_DIR``
     - Parent output directory
   * - ``MAX_INFERENCE_SAMPLES``
     - Limit inference to the first N JSONL rows (useful for smokes)
   * - ``RESUME=1``
     - Continue a partially written run from ``rank_results/``; use the same config, ``INFERENCE_NAME``, and number of ranks as the original job

AutoModel Fields
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Role
   * - ``model_name``
     - Must be ``nemotron_omni_hf`` (for the Nemotron-3-Nano-Omni model)
   * - ``model_path``
     - HF model id or local path to base weights (for LoRA, must be the **base** model, not a consolidated SFT tree)
   * - ``is_lora``
     - ``true`` when running with adapter weights only; ``false`` for base or full SFT
   * - ``adapter_path``
     - Directory with ``adapter_config.json`` and ``adapter_model.safetensors`` (required when ``is_lora: true``)

Megatron-Bridge Fields
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Role
   * - ``model_name``
     - Must be ``nemotron_omni_mbridge`` (for the Nemotron-3-Nano-Omni model)
   * - ``hf_model_path``
     - HF model id used for the processor/tokenizer
   * - ``model_path``
     - Megatron checkpoint directory (for example ``.../iter_0003100``)
   * - ``tensor_model_parallel_size``
     - Tensor parallelism; must match training
   * - ``pipeline_model_parallel_size``
     - Pipeline parallelism; must match training
   * - ``context_parallel_size``
     - Context parallelism; must match training
   * - ``expert_model_parallel_size``
     - Expert parallelism (EP); shards which MoE experts live on which GPUs. Must match training ``ep_size``.
   * - ``expert_tensor_parallel_size``
     - Expert tensor parallelism (ETP); shards each expert's weights across GPUs (within-expert TP), separate from ``tensor_model_parallel_size``. Usually ``1`` unless training used ETP > 1.
   * - ``resize``
     - When ``true``, resize video frames to 512×512 before encoding (Bridge path)

LoRA Inference (Adapter-Only Weights)
-------------------------------------

LoRA training saves **adapter weights only** — not a full copy of the 30B base model. Inference therefore always needs the **base model** plus the adapter, or a **merged** checkpoint produced offline.

AutoModel
~~~~~~~~~

Use ``default_lora.yaml`` with:

- ``model_path`` — HF base model (for example ``nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16``). Do **not** point this at a consolidated SFT checkpoint.
- ``is_lora: true``
- ``adapter_path`` — training output directory containing ``adapter_config.json`` and ``adapter_model.safetensors`` (often under ``.../checkpoints_.../epoch_*_step_*`` or a ``model/`` subfolder)

At load time the driver reads the LoRA matrices, maps wrapper layer names to the HF model, and **adds the low-rank delta directly into the base weight tensors** in memory. You do not need to merge adapters to disk before inference, but you do need the base model weights available locally or on HuggingFace Hub.

.. code-block:: bash

   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_lora.yaml \
   ADAPTER_PATH=avlm/training/automodel/lora/slurm/outputs/<MODEL_NAME>/checkpoints_.../epoch_9_step_429 \
   bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh

Megatron-Bridge
~~~~~~~~~~~~~~~

Bridge inference loads a **full Megatron checkpoint** from ``model_path``. A raw LoRA ``iter_*`` directory contains adapter deltas only and cannot be used directly with ``default_mbridge.yaml``.

Two options:

1. **Merge LoRA into Megatron weights** with ``merge_lora.py`` under ``${MEGATRON_BRIDGE_ROOT}/examples/peft/``, then set ``model_path`` to the merged ``iter_*`` directory. See `megatron-bridge/lora/LORA_GUIDE.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/training/megatron-bridge/lora/LORA_GUIDE.MD>`_ (LoRA Merge section).
2. **Export an HF PEFT adapter** with ``export_adapter.py``, then run inference through **AutoModel** ``default_lora.yaml`` as above.

Outputs
-------

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl
   avlm/inference/outputs/<INFERENCE_NAME>/predictions.json

Multi-GPU runs also write per-rank scratch files under ``rank_results/`` until all ranks finish and rank 0 merges.

For MCQ scoring and LLM-judge evaluation of these predictions, see :doc:`../evaluation`.
