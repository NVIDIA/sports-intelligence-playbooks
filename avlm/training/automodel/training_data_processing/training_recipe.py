# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM finetune entry point with HF conversation JSONL (SFT and LoRA).

Use in YAML (under ``dataloader``)::

    dataloader:
      use_sequence_packing: true
      packed_sequence:
        pack_size: 24576
        ...
      collate_fn: ...   # used when use_sequence_packing is false

AutoModel still reads a top-level ``packed_sequence`` at runtime; the recipe
hoists ``dataloader.packed_sequence`` before training when packing is enabled.
"""

from __future__ import annotations

import logging
import math
import os
import time
from copy import deepcopy
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, filter_forward_kwargs
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM as _FinetuneRecipeForVLM
from nemo_automodel.recipes.vlm import finetune as finetune_mod
from nemo_automodel.recipes.vlm.finetune import _get_model_name, calculate_loss, main as _main
from avlm.training.automodel.training_data_processing.val_shard_sampler import (
    ValShardSampler,
    log_val_shard_layout,
)
from avlm.utils.batch_memory import (
    post_optimizer_cpu_gc,
    release_grad_accum_batches,
    release_vlm_batch,
)

logger = logging.getLogger(__name__)


_VALID_WANDB_MODES = frozenset({"online", "offline", "disabled"})


def _normalize_wandb_mode(raw: Any) -> str | None:
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if not mode:
        return None
    if mode not in _VALID_WANDB_MODES:
        raise ValueError(
            f"Invalid wandb.mode={raw!r}; expected one of: {', '.join(sorted(_VALID_WANDB_MODES))}"
        )
    return mode


def _set_wandb_cfg_mode(wandb_cfg: Any, mode: str) -> None:
    if hasattr(wandb_cfg, "mode"):
        wandb_cfg.mode = mode
    elif isinstance(wandb_cfg, dict):
        wandb_cfg["mode"] = mode


def _apply_wandb_mode(cfg: Any) -> None:
    """Apply wandb.mode from recipe YAML (or CLI) and sync WANDB_MODE before wandb.init."""
    wandb_cfg = _cfg_get(cfg, "wandb", None)
    if wandb_cfg is None:
        return

    env_mode = _normalize_wandb_mode(os.environ.get("WANDB_MODE") or None)
    yaml_mode = _normalize_wandb_mode(_cfg_get(wandb_cfg, "mode", None))

    if env_mode:
        mode = env_mode
    elif yaml_mode:
        mode = yaml_mode
    else:
        mode = "online"

    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        logger.info("WANDB_API_KEY unset; falling back to WANDB_MODE=offline")
        mode = "offline"
    elif yaml_mode and not env_mode:
        logger.info("Using wandb.mode=%s from recipe config", mode)

    os.environ["WANDB_MODE"] = mode
    _set_wandb_cfg_mode(wandb_cfg, mode)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if hasattr(node, "to_dict"):
        return deepcopy(node.to_dict())
    if isinstance(node, dict):
        return deepcopy(node)
    return {k: v for k, v in node.__dict__.items() if not k.startswith("_")}


def dataloader_packed_sequence_cfg(cfg: Any) -> Any:
    """Train ``packed_sequence`` from ``dataloader.packed_sequence`` (legacy: top-level)."""
    dl = _cfg_get(cfg, "dataloader", None)
    if dl is not None:
        ps = _cfg_get(dl, "packed_sequence", None)
        if ps is not None:
            return ps
    return _cfg_get(cfg, "packed_sequence", None)


def validation_packed_sequence_overrides(cfg: Any) -> Any:
    """Val-only packing overrides from ``validation_dataloader.packed_sequence``."""
    val_dl = _cfg_get(cfg, "validation_dataloader", None)
    return _cfg_get(val_dl, "packed_sequence", None) if val_dl is not None else None


def sequence_packing_enabled(cfg: Any) -> bool:
    """True when ``dataloader.use_sequence_packing`` is set (else inferred from ``pack_size``)."""
    dl = _cfg_get(cfg, "dataloader", None)
    if dl is not None:
        flag = _cfg_get(dl, "use_sequence_packing", None)
        if flag is not None:
            return bool(flag)
    flag = _cfg_get(cfg, "use_sequence_packing", None)
    if flag is not None:
        return bool(flag)
    ps = dataloader_packed_sequence_cfg(cfg)
    if ps is None:
        return False
    return int(getattr(ps, "pack_size", 0) or 0) > 0


def packed_sequence_pack_size(cfg: Any) -> int:
    ps = dataloader_packed_sequence_cfg(cfg)
    if ps is None:
        ps = _cfg_get(cfg, "packed_sequence", None)
    if ps is None:
        return 0
    return int(getattr(ps, "pack_size", 0) or 0)


def _config_mutation_root(cfg: Any) -> Any:
    """Return the ``ConfigNode`` that ``RecipeConfig.get()`` and ``build_dataloader`` read."""
    if hasattr(cfg, "_raw"):
        return cfg._raw
    return cfg


def hoist_packed_sequence_for_automodel(cfg: Any, *, enabled: bool) -> None:
    """Copy ``dataloader.packed_sequence`` to top-level ``packed_sequence`` for AutoModel."""
    root = _config_mutation_root(cfg)
    if enabled:
        ps = dataloader_packed_sequence_cfg(cfg)
        if ps is not None:
            root.packed_sequence = ConfigNode(_cfg_to_dict(ps))
    else:
        root.packed_sequence = ConfigNode({"pack_size": 0, "max_length": 0})
    strip_dataloader_packing_keys(cfg)


def strip_dataloader_packing_keys(cfg: Any) -> None:
    """Drop packing metadata from dataloader configs before ``StatefulDataLoader`` init.

    AutoModel reads top-level ``packed_sequence``; keys under ``dataloader`` are recipe-only
    and must not be forwarded to ``cfg_dl.instantiate()``.
    """
    for dl_key in ("dataloader", "validation_dataloader"):
        dl = _cfg_get(cfg, dl_key, None)
        if dl is None:
            continue
        for key in ("use_sequence_packing", "packed_sequence"):
            if key in dl.__dict__:
                del dl.__dict__[key]


def _env_int(*names: str) -> int:
    for name in names:
        raw = os.environ.get(name, "")
        if raw:
            return int(raw)
    return 0


def _wrap_robust_collate(dataloader: Any, *, max_retries: int = 10) -> None:
    """Re-sample batches when collate rejects overlong or malformed samples."""
    from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

    dataset = getattr(dataloader, "dataset", None)
    collate_fn = getattr(dataloader, "collate_fn", None)
    if dataset is None or collate_fn is None:
        return
    dataloader.collate_fn = make_robust_collate(dataset, collate_fn, max_retries=max_retries)


def _enable_media_activation_checkpointing(model: Any) -> None:
    """Enable native recomputation for trainable RADIO and Parakeet towers."""
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and any(param.requires_grad for param in vision_model.parameters()):
        radio = getattr(getattr(vision_model, "radio_model", None), "model", None)
        if radio is not None:
            set_checkpointing = getattr(radio, "set_grad_checkpointing", None)
            if callable(set_checkpointing):
                set_checkpointing(True)
            else:
                radio.grad_checkpointing = True
            logger.info("Enabled activation checkpointing for the RADIO vision tower")

    sound_encoder = getattr(model, "sound_encoder", None)
    if sound_encoder is not None and any(param.requires_grad for param in sound_encoder.parameters()):
        enable_checkpointing = getattr(sound_encoder, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
            logger.info("Enabled activation checkpointing for the Parakeet sound tower")


class _LimitedDataLoader:
    """Wrap a dataloader and yield at most ``max_batches`` batches."""

    def __init__(self, dataloader: Any, max_batches: int) -> None:
        self._dataloader = dataloader
        self._max_batches = max_batches

    def __iter__(self) -> Iterator[Any]:
        for batch_idx, batch in enumerate(self._dataloader):
            if batch_idx >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(self._max_batches, len(self._dataloader))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)


def _val_dp_shard_info(device_mesh: Any) -> tuple[int, int]:
    """Return ``(dp_size, dp_rank)`` for validation sharding."""
    if device_mesh is None:
        return 1, 0
    from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

    dp_mesh = get_flat_mesh(device_mesh, "dp")
    return dp_mesh.size(), dp_mesh.get_local_rank()


def _to_plain_val_dataloader(dataloader: Any) -> DataLoader:
    """Use a plain ``DataLoader`` so each validation pass re-reads the val set."""
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError:
        StatefulDataLoader = ()  # type: ignore[misc, assignment]

    if isinstance(dataloader, DataLoader) and not isinstance(dataloader, StatefulDataLoader):
        return dataloader

    if isinstance(dataloader, StatefulDataLoader):
        logger.info("Validation uses a plain DataLoader (full re-read on each val pass).")
        return DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=dataloader.sampler,
            num_workers=getattr(dataloader, "num_workers", 0),
            collate_fn=dataloader.collate_fn,
            pin_memory=getattr(dataloader, "pin_memory", False),
            drop_last=getattr(dataloader, "drop_last", False),
            timeout=getattr(dataloader, "timeout", 0),
            worker_init_fn=getattr(dataloader, "worker_init_fn", None),
        )

    return dataloader


def _finalize_val_dataloader(dataloader: Any, *, num_replicas: int, rank: int) -> DataLoader:
    """Plain, repeatable val loader with one example per rank at most (no pad duplicates)."""
    plain = _to_plain_val_dataloader(dataloader)
    num_examples = len(plain.dataset)
    log_val_shard_layout(num_examples, num_replicas, rank)

    return DataLoader(
        dataset=plain.dataset,
        batch_size=plain.batch_size,
        sampler=ValShardSampler(plain.dataset, num_replicas, rank),
        num_workers=getattr(plain, "num_workers", 0),
        collate_fn=plain.collate_fn,
        pin_memory=getattr(plain, "pin_memory", False),
        drop_last=False,
        timeout=getattr(plain, "timeout", 0),
        worker_init_fn=getattr(plain, "worker_init_fn", None),
    )


class FinetuneRecipeForVLM(_FinetuneRecipeForVLM):
    """VLM finetune recipe; optional neat packing via ``dataloader.use_sequence_packing``."""

    def __init__(self, cfg):
        # Hoist before RecipeConfig wraps cfg: launcher CLI overrides may replace
        # top-level packed_sequence with size-only keys (dropping packing_ratio).
        self._validation_packed_sequence_overrides = _cfg_to_dict(
            validation_packed_sequence_overrides(cfg)
        )
        enabled = sequence_packing_enabled(cfg) and packed_sequence_pack_size(cfg) > 0
        hoist_packed_sequence_for_automodel(cfg, enabled=enabled)
        super().__init__(cfg)

    def _sequence_packing_active(self) -> bool:
        if not sequence_packing_enabled(self.cfg):
            return False
        return packed_sequence_pack_size(self.cfg) > 0 or int(
            getattr(_cfg_get(self.cfg, "packed_sequence", None), "pack_size", 0) or 0
        ) > 0

    def _validation_packed_sequence_cfg(self, cfg_ps: Any) -> Any:
        """Return train ``packed_sequence`` merged with val-only overrides."""
        if cfg_ps is None:
            return None
        if not self._validation_packed_sequence_overrides:
            return cfg_ps
        if hasattr(cfg_ps, "raw_config"):
            merged = deepcopy(cfg_ps.raw_config)
        else:
            merged = dict(cfg_ps)
        merged.update(deepcopy(self._validation_packed_sequence_overrides))
        return ConfigNode(merged)

    def _forward_backward_step(self, idx, batch, **kwargs):
        """Release the CPU collated batch after each grad-accum microbatch."""
        try:
            super()._forward_backward_step(idx, batch, **kwargs)
        finally:
            if kwargs.get("is_train", True):
                release_vlm_batch(batch)

    def _run_train_optim_step(self, batches, max_grad_norm: float | None = None):
        try:
            return super()._run_train_optim_step(batches, max_grad_norm)
        finally:
            release_grad_accum_batches(batches)
            post_optimizer_cpu_gc()

    def run_train_validation_loop(self):
        """Training loop with CPU batch cleanup after every optimizer step."""
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        pbar = self._make_progress_bar()
        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)
                for _batch_idx, batches in enumerate(self.step_scheduler):
                    try:
                        log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                        self.log_train_metrics(log_data)
                        self._update_progress_bar(pbar, log_data.metrics)

                        val_loss = {}
                        if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                            if self.pp_enabled:
                                logger.warning("Validation is not supported for pipeline parallelism")
                            else:
                                val_log_data = self._run_validation_epoch(self.val_dataloader)
                                val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                                self.log_val_metrics(val_log_data)
                            for mp in self.model_parts:
                                mp.train()

                        if self.step_scheduler.is_ckpt_step:
                            self.save_checkpoint(
                                epoch,
                                self.step_scheduler.step,
                                log_data.metrics["loss"],
                                val_loss,
                                best_metric_key=self.best_metric_key,
                            )
                        self._maybe_collect_garbage()
                    finally:
                        release_grad_accum_batches(batches)
                        post_optimizer_cpu_gc()
        finally:
            if pbar is not None:
                pbar.close()

        self.metric_logger_train.close()
        self.metric_logger_valid.close()

        self.checkpointer.close()

        if self.step_scheduler.sigterm_flag:
            from nemo_automodel.components.loggers.mlflow_utils import end_mlflow_active_run_as_killed

            end_mlflow_active_run_as_killed()

    def setup(self):
        """Build components; apply packing patches and packed val when enabled."""
        from avlm.utils.sigint_handler import register_interactive_sigint_handler

        register_interactive_sigint_handler()
        _apply_wandb_mode(self.cfg)
        packing_enabled = sequence_packing_enabled(self.cfg) and packed_sequence_pack_size(self.cfg) > 0
        if packing_enabled:
            from avlm.training.automodel.training_data_processing.packing import apply_video_sound_packing_patches

            apply_video_sound_packing_patches()
            logger.info("dataloader.use_sequence_packing=true: applied video-sound neat-packing patches.")
        super().setup()
        # Upstream nemotron_omni logs sound-token shapes at INFO every forward; mute after model build.
        if os.environ.get("AVLM_VERBOSE_OMNI_LOGS", "0") != "1":
            logging.getLogger("nemo_automodel.components.models.nemotron_omni.model").setLevel(logging.WARNING)
        if getattr(self, "activation_checkpointing", False):
            for model_part in self.model_parts:
                _enable_media_activation_checkpointing(model_part)
        _wrap_robust_collate(self.dataloader)
        if self.val_dataloader is not None:
            _wrap_robust_collate(self.val_dataloader)
        if packing_enabled:
            logger.info(
                "dataloader.use_sequence_packing=true: robust collate retries overlong packed batches."
            )
        else:
            logger.info(
                "dataloader.use_sequence_packing=false: collate skips samples longer than max_length."
            )
        skip_val = os.environ.get("SKIP_VALIDATION")
        if skip_val == "1":
            self.val_dataloader = None
            logger.info("SKIP_VALIDATION=1: validation disabled for this run")
            return

        if packing_enabled and "validation_dataset" in self.cfg:
            self.val_dataloader, _ = self._build_val_dataloader()
            cfg_ps = self._validation_packed_sequence_cfg(
                _cfg_get(self.cfg, "packed_sequence", None) or dataloader_packed_sequence_cfg(self.cfg)
            )
            if cfg_ps is not None and getattr(cfg_ps, "pack_size", 0) > 0:
                logger.info(
                    "Validation dataloader uses packed_sequence (pretokenize=%s, pack_size=%s, max_packs=%s)",
                    getattr(cfg_ps, "pretokenize", True),
                    getattr(cfg_ps, "pack_size", None),
                    getattr(cfg_ps, "max_packs", None),
                )
        elif self.val_dataloader is not None:
            self.val_dataloader = self._finalize_val_dataloader(self.val_dataloader)

    def _finalize_val_dataloader(self, dataloader: Any) -> DataLoader:
        num_replicas, rank = _val_dp_shard_info(self.device_mesh)
        return _finalize_val_dataloader(dataloader, num_replicas=num_replicas, rank=rank)

    def _build_val_dataloader(self):
        """Build validation ``StatefulDataLoader`` (packed when sequence packing is enabled)."""
        if "validation_dataset" not in self.cfg:
            raise ValueError("validation_dataset is missing from config")

        get_rope_index = getattr(self.model_parts[0], "get_rope_index", None)
        pp_n_microbatches = None
        pp_cp_preembed = (
            self.pp_enabled
            and self.dist_setup.cp_size > 1
            and hasattr(self.model_parts[0], "prepare_model_inputs_for_cp")
        )
        if self.pp_enabled and not pp_cp_preembed:
            pp_n_microbatches = self.pp.pp_batch_size // self.pp.pp_microbatch_size

        packing_enabled = sequence_packing_enabled(self.cfg) and packed_sequence_pack_size(self.cfg) > 0
        cfg_ps = None
        if packing_enabled:
            cfg_ps = self._validation_packed_sequence_cfg(
                _cfg_get(self.cfg, "packed_sequence", None) or dataloader_packed_sequence_cfg(self.cfg)
            )

        dl, processor = finetune_mod.build_dataloader(
            self.cfg.validation_dataset,
            self.cfg.validation_dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            cfg_model=self.cfg.model,
            cfg_ps=cfg_ps,
            get_rope_index=get_rope_index,
            pp_n_microbatches=pp_n_microbatches,
        )
        return self._finalize_val_dataloader(dl), processor

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run validation with a repeatable loader aligned across DP ranks for MoE."""
        max_batches = _env_int("VAL_MAX_BATCHES")
        if max_batches > 0:
            logger.info("Partial validation epoch capped at %d batches", max_batches)
            val_dataloader = _LimitedDataLoader(val_dataloader, max_batches)

        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            skipped_unsupervised = 0
            for batch in val_dataloader:
                batch = {
                    k: (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                labels_tensor = batch.get("labels")
                if labels_tensor is None:
                    skipped_unsupervised += 1
                    continue
                num_label_tokens = (labels_tensor != -100).sum().item()
                score_loss = num_label_tokens > 0
                if not score_loss:
                    skipped_unsupervised += 1

                _model = self.model_parts[0]
                _cp_active = (
                    self.device_mesh is not None
                    and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
                    and self.device_mesh["cp"].size() > 1
                    and not self.pp_enabled
                )
                if _cp_active and hasattr(_model, "prepare_model_inputs_for_cp"):
                    mm_kwargs = {k: batch[k] for k in VLM_INPUT_KEYS if batch.get(k) is not None}
                    with torch.no_grad():
                        prepared = _model(_pre_embed_only=True, **mm_kwargs)
                    for k in VLM_INPUT_KEYS:
                        batch.pop(k, None)
                    batch.update(prepared)

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
                labels = batch.pop("labels")
                with train_ctx():
                    batch = filter_forward_kwargs(self.model_parts[0], batch)
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model_parts[0](logits_to_keep=1, **batch)
                    else:
                        out = self.model_parts[0](**batch)
                    if score_loss:
                        local_loss = calculate_loss(
                            self.loss_fn,
                            logits=getattr(out, "logits", out),
                            labels=labels,
                            model=self.model_parts[0],
                            hidden_states=out.hidden_states[-1]
                            if getattr(out, "hidden_states", None) is not None
                            else None,
                            num_label_tokens=num_label_tokens,
                        )
                        local_loss = self._maybe_add_drafter_loss(
                            out=out,
                            base_loss=local_loss,
                            labels=labels,
                            model=self.model_parts[0],
                            num_label_tokens=num_label_tokens,
                        )
                        total_num_label_tokens += num_label_tokens
                        total_loss += local_loss.item() * num_label_tokens
                        total_tokens += num_label_tokens

        if skipped_unsupervised:
            logger.warning(
                "Validation ran %d batch(es) without supervised label tokens (forward only, MoE sync)",
                skipped_unsupervised,
            )

        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        if total_num_label_tokens == 0:
            logger.error(
                "Validation produced zero supervised label tokens globally; val_loss is undefined"
            )
            val_loss = float("nan")
        else:
            val_loss = total_loss / total_tokens

        if total_num_label_tokens > 0 and (math.isnan(val_loss) or math.isinf(val_loss)):
            logger.error(
                "Validation loss is non-finite despite %d label tokens (loss=%s)",
                total_num_label_tokens,
                val_loss,
            )

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )


def main(config_path=None):
    """Delegate to stock VLM finetune ``main`` (packing patches apply in ``setup`` when enabled)."""
    return _main(config_path)


if __name__ == "__main__":
    main()
