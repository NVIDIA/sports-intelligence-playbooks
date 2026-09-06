# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installed into Megatron-Bridge as:
#   src/megatron/bridge/recipes/nemotron_omni/recipe.py

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.recipes.nemotron_omni.nemotron_omni import _DEFAULT_HF_PATH, _nemotron_omni_base_config
from megatron.bridge.training.config import ConfigContainer, DatasetBuildContext
from megatron.bridge.training.mixed_precision import get_mixed_precision_config

from .collate import conversation_jsonl_collate_fn
from .dataset import load_jsonl_examples, normalize_jsonl_format, plan_sequence_packs


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true"}


def _load_processor(hf_processor_path: str, *, trust_remote_code: bool) -> Any:
    return AutoProcessor.from_pretrained(
        hf_processor_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_processor_path,
        ),
    )


def _build_jsonl_dataset(
    *,
    jsonl_path: str,
    media_root: str,
    jsonl_format: str,
    target_length: int,
    processor: Any,
    collate_impl: Callable[..., dict],
    seq_length: int,
    pack_sequences: bool,
    capacity_pack_sequences: bool,
    video_metadata_path: str | None,
    packing_ratio: float,
    max_video_frames: int,
    video_sample_fps: float,
    max_samples: int | None,
    sample_ratio: float | None,
    rng_seed: int,
) -> VLMConversationDataset | None:
    if not jsonl_path or target_length <= 0:
        return None
    examples = load_jsonl_examples(
        jsonl_path,
        media_root,
        jsonl_format=normalize_jsonl_format(jsonl_format),
        max_samples=max_samples,
        sample_ratio=sample_ratio,
        rng_seed=rng_seed,
        video_metadata_path=video_metadata_path if capacity_pack_sequences else None,
    )
    if capacity_pack_sequences:
        examples = plan_sequence_packs(
            examples,
            seq_length=seq_length,
            packing_ratio=packing_ratio,
            max_video_frames=max_video_frames,
            video_sample_fps=video_sample_fps,
        )
    return VLMConversationDataset(
        base_examples=examples,
        target_length=target_length,
        processor=processor,
        collate_impl=collate_impl,
        pack_sequences=pack_sequences or capacity_pack_sequences,
    )


@dataclass(kw_only=True)
class ConversationJsonlProvider(PreloadedVLMConversationProvider):
    """JSONL LLaVA / HF conversations with Omni temporal-video task-encoder collate."""

    train_jsonl: str = ""
    val_jsonl: str = ""
    video_root: str = ""
    val_video_root: str = ""
    train_jsonl_format: str = "llava"
    val_jsonl_format: str | None = None
    max_video_frames: int = 128
    video_sample_fps: float = 2.0
    val_max_video_frames: int | None = None
    val_video_sample_fps: float | None = None
    train_sample_ratio: float | None = None
    val_sample_ratio: float | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    rng_seed: int = 42
    video_metadata_path: str | None = None
    packing_ratio: float = 1.0
    capacity_pack_sequences: bool = False
    collate_impl: Optional[Callable[..., dict]] = None

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        if not self.train_jsonl or not self.video_root:
            raise ValueError("dataset.train_jsonl and dataset.video_root must be set for JSONL conversation SFT")

        processor = _load_processor(self.hf_processor_path, trust_remote_code=self.trust_remote_code)
        collate = self.collate_impl or partial(
            conversation_jsonl_collate_fn,
            seq_length=self.seq_length,
            max_video_frames=self.max_video_frames,
            video_sample_fps=self.video_sample_fps,
        )
        train_ds = _build_jsonl_dataset(
            jsonl_path=self.train_jsonl,
            media_root=self.video_root,
            jsonl_format=self.train_jsonl_format,
            target_length=context.train_samples,
            processor=processor,
            collate_impl=collate,
            seq_length=self.seq_length,
            pack_sequences=self.pack_sequences_in_batch,
            capacity_pack_sequences=self.capacity_pack_sequences,
            video_metadata_path=self.video_metadata_path,
            packing_ratio=self.packing_ratio,
            max_video_frames=self.max_video_frames,
            video_sample_fps=self.video_sample_fps,
            max_samples=self.max_train_samples,
            sample_ratio=self.train_sample_ratio,
            rng_seed=self.rng_seed,
        )
        val_mvf = self.val_max_video_frames if self.val_max_video_frames is not None else self.max_video_frames
        val_vsf = self.val_video_sample_fps if self.val_video_sample_fps is not None else self.video_sample_fps
        val_collate = partial(
            conversation_jsonl_collate_fn,
            seq_length=self.seq_length,
            max_video_frames=val_mvf,
            video_sample_fps=val_vsf,
            processed_prompt_split="val",
        )
        valid_ds = _build_jsonl_dataset(
            jsonl_path=self.val_jsonl,
            media_root=self.val_video_root or self.video_root,
            jsonl_format=self.val_jsonl_format or self.train_jsonl_format,
            target_length=context.valid_samples,
            processor=processor,
            collate_impl=val_collate,
            seq_length=self.seq_length,
            pack_sequences=False,
            capacity_pack_sequences=False,
            video_metadata_path=None,
            packing_ratio=1.0,
            max_video_frames=val_mvf,
            video_sample_fps=val_vsf,
            max_samples=self.max_val_samples,
            sample_ratio=self.val_sample_ratio,
            rng_seed=self.rng_seed,
        )
        return train_ds, valid_ds, None


def _conversation_jsonl_model_defaults(cfg: ConfigContainer) -> None:
    """Temporal video + audio SFT defaults (YAML / CLI overrides win)."""
    cfg.model.dynamic_resolution = True
    cfg.model.moe_aux_loss_coeff = 0
    cfg.model.temporal_patch_dim = 2
    cfg.model.separate_video_embedder = True
    cfg.model.temporal_ckpt_compat = True
    cfg.model.moe_router_enable_expert_bias = True
    cfg.model.moe_router_bias_update_rate = 0.0
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
    cfg.model.radio_interpolate_only_cpe = False

    if _env_enabled("BF16_OPTIMIZER_STATES"):
        cfg.mixed_precision = get_mixed_precision_config("bf16_mixed")
        cfg.optimizer.exp_avg_dtype = torch.bfloat16
        cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    if _env_enabled("USE_MEGATRON_FSDP"):
        cfg.dist.use_megatron_fsdp = True
        cfg.ddp.use_megatron_fsdp = True
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        cfg.ddp.average_in_collective = False
        cfg.checkpoint.ckpt_format = "fsdp_dtensor"
        cfg.ddp.megatron_fsdp_main_params_dtype = None


def conversation_jsonl_sft_config(hf_path: str = _DEFAULT_HF_PATH) -> ConfigContainer:
    """SFT on JSONL conversations (llava or hf format)."""
    cfg = _nemotron_omni_base_config(hf_path=hf_path)
    _conversation_jsonl_model_defaults(cfg)
    cfg.dataset = ConversationJsonlProvider(
        seq_length=cfg.model.seq_length,
        hf_processor_path=hf_path,
        pack_sequences_in_batch=False,
        num_workers=4,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
    )
    return cfg


def conversation_jsonl_peft_config(hf_path: str = _DEFAULT_HF_PATH) -> ConfigContainer:
    """LoRA PEFT on JSONL conversations."""
    from megatron.bridge.peft.lora import LoRA

    cfg = conversation_jsonl_sft_config(hf_path=hf_path)
    if _env_enabled("USE_MEGATRON_FSDP"):
        cfg.ddp.overlap_grad_reduce = True
        cfg.ddp.bucket_size = 64_000_000
    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "in_proj", "out_proj"],
        dim=16,
        alpha=32,
    )
    cfg.checkpoint.load = None
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True
    return cfg
