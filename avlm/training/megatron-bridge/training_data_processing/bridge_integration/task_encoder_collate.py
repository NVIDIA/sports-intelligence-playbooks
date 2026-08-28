# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL collate and LLaVA hooks for Nemotron Omni in-batch sequence packing.

Collate: HF preprocessing adapted to MCore + per-segment TP pad + unpacked MBS>1 vision cat.
Forward hooks (packed only): rebuild ``cu_seqlens`` in combined embedding space after
``LLaVAModel._preprocess_data`` expands vision tokens. Unpacked batches are unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

from .hf_mcore_task_encoder import HfMcoreNemotronOmniTaskEncoder
from .processed_prompt_log import log_encoded_sample
from ..video_utils import prepare_collate_example

logger = logging.getLogger("avlm.pack")
_pack_log_count = 0
_IGNORE_INDEX = -100


def _pack_pad_multiple() -> int:
    return max(1, int(os.environ.get("AVLM_PACK_PAD_MULTIPLE", "2")))


def _align_packed_segments_for_parallel(out: dict[str, Any], *, pad_multiple: int) -> dict[str, Any]:
    """Per-segment text TP alignment (vlm_step-style) before combined-space cu rebuild in forward."""
    if pad_multiple <= 1 or out.get("cu_seqlens") is None:
        return out
    ids = out.get("input_ids")
    if ids is None or ids.dim() != 2 or ids.size(0) != 1:
        return out
    labels, loss_mask, pos = out.get("labels"), out.get("loss_mask"), out.get("position_ids")
    device = ids.device
    cu_in = [int(x) for x in out["cu_seqlens"].tolist()]
    if len(cu_in) <= 1:
        return out

    ids_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    loss_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    seg_lens: list[int] = []
    new_cu = [0]

    for i in range(len(cu_in) - 1):
        start_i, end_i = cu_in[i], cu_in[i + 1]
        seg_len = end_i - start_i
        pad_len = (-seg_len) % pad_multiple
        seg_lens.append(seg_len + pad_len)
        ids_parts.append(ids[0, start_i:end_i])
        if labels is not None:
            labels_parts.append(labels[0, start_i:end_i])
        if loss_mask is not None:
            loss_parts.append(loss_mask[0, start_i:end_i])
        if pos is not None:
            pos_parts.append(pos[0, start_i:end_i])
        if pad_len:
            ids_parts.append(torch.zeros(pad_len, dtype=ids.dtype, device=device))
            if labels is not None:
                labels_parts.append(
                    torch.full((pad_len,), _IGNORE_INDEX, dtype=labels.dtype, device=device)
                )
            if loss_mask is not None:
                loss_parts.append(torch.zeros(pad_len, dtype=loss_mask.dtype, device=device))
            if pos is not None:
                last = int(pos[0, end_i - 1].item()) if seg_len else -1
                pos_parts.append(
                    torch.arange(last + 1, last + 1 + pad_len, device=device, dtype=pos.dtype)
                )
        new_cu.append(new_cu[-1] + seg_len + pad_len)

    out["input_ids"] = torch.cat(ids_parts, dim=0).unsqueeze(0)
    if out.get("tokens") is not None:
        out["tokens"] = out["input_ids"]
    if labels is not None:
        out["labels"] = torch.cat(labels_parts, dim=0).unsqueeze(0)
    if loss_mask is not None:
        out["loss_mask"] = torch.cat(loss_parts, dim=0).unsqueeze(0)
    if pos is not None:
        out["position_ids"] = torch.cat(pos_parts, dim=0).unsqueeze(0)
    cu_t = torch.tensor(new_cu, dtype=out["cu_seqlens"].dtype, device=out["cu_seqlens"].device)
    out["cu_seqlens"] = cu_t
    if out.get("cu_seqlens_unpadded") is not None:
        out["cu_seqlens_unpadded"] = cu_t.clone()
    if out.get("max_seqlen") is not None:
        out["max_seqlen"] = torch.tensor(max(seg_lens), dtype=out["max_seqlen"].dtype)
    return out


def _encoder_seq_length(
    seq_length: int,
    *,
    pack_sequences: bool,
    num_samples: int,
    capacity_planned: bool,
) -> int:
    if pack_sequences and num_samples > 1 and not capacity_planned:
        return max(1, seq_length // num_samples)
    return seq_length


def _merge_unpacked_visual_batch(batch, samples):  # type: ignore[no-untyped-def]
    visual_tensors: dict[str, torch.Tensor | None] = {}
    keys = {k for s in samples for k in s.visual_tensors}
    for key in keys:
        tensors = [s.visual_tensors[key] for s in samples if key in s.visual_tensors]
        visual_tensors[key] = torch.cat(tensors, dim=1) if tensors else None
    return dataclasses.replace(batch, visual_tensors=visual_tensors)


class _AvlmHfMcoreTaskEncoder(HfMcoreNemotronOmniTaskEncoder):
    """HF-MCore encoder with unpacked MBS>1 visual batching."""

    def batch(self, samples):  # type: ignore[override]
        if self.pack_sequences or len(samples) <= 1:
            return super().batch(samples)
        batch = super().batch(
            [dataclasses.replace(s, visual_tensors={}) for s in samples]
        )
        return _merge_unpacked_visual_batch(batch, samples)


def make_task_encoder(
    processor,
    *,
    seq_length: int,
    pack_sequences: bool = False,
) -> HfMcoreNemotronOmniTaskEncoder:
    return _AvlmHfMcoreTaskEncoder(
        processor=processor,
        seq_length=seq_length,
        num_mel_bins=128,
        patch_dim=16,
        pack_sequences=pack_sequences,
    )


def _conversation_json(conversation: list) -> str:
    cleaned: list[dict[str, Any]] = []
    for message in conversation:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                {k: v for k, v in item.items() if not str(k).startswith("_") and k != "metadata"}
                if isinstance(item, dict)
                else item
                for item in content
            ]
        cleaned.append(msg)
    return json.dumps(cleaned)


def prepared_to_chatml_sample(prepared: dict[str, Any], *, key: str = "0") -> ChatMLSample:
    frames: list[Image.Image] | None = prepared.get("_avlm_video_frames")
    audio: torch.Tensor | None = None
    if prepared.get("audio") is not None:
        array, _sr = prepared["audio"]
        audio = torch.from_numpy(np.asarray(array, dtype=np.float32))

    base_fields = {f.name for f in dataclasses.fields(ChatMLSample)}
    kwargs: dict[str, Any] = {"conversation": _conversation_json(prepared["conversation"])}
    if "__key__" in base_fields:
        kwargs["__key__"] = key
    if "__subflavors__" in base_fields:
        subflavors: dict[str, Any] = {}
        source_fps = prepared.get("_avlm_video_source_fps")
        if source_fps:
            subflavors["video_source_fps"] = float(source_fps)
        indices = prepared.get("_avlm_video_frame_indices")
        if indices:
            subflavors["video_frame_indices"] = [int(i) for i in indices]
        kwargs["__subflavors__"] = subflavors
    if "__restore_key__" in base_fields:
        kwargs["__restore_key__"] = ()
    if "__subflavor__" in base_fields:
        kwargs["__subflavor__"] = None
    if frames is not None:
        kwargs["videos"] = frames
    if audio is not None:
        kwargs["audio"] = audio
    return ChatMLSample(**kwargs)


def _prepared_sample_key(prepared: dict[str, Any], *, batch_index: int) -> str:
    for field in ("id", "sample_id", "key", "__key__"):
        if prepared.get(field) is not None:
            return str(prepared[field])
    for field in ("video-sound", "video", "audio_video_path"):
        if prepared.get(field):
            return str(prepared[field])
    return str(batch_index)


def collate_prepared_via_task_encoder(
    prepared: list[dict[str, Any]],
    processor,
    *,
    seq_length: int,
    pack_sequences: bool,
    capacity_planned: bool = False,
    sample_keys: list[str] | None = None,
    processed_prompt_split: str = "train",
) -> dict[str, Any]:
    packing = pack_sequences and len(prepared) > 1
    encoder = make_task_encoder(
        processor,
        seq_length=_encoder_seq_length(
            seq_length,
            pack_sequences=packing,
            num_samples=len(prepared),
            capacity_planned=capacity_planned,
        ),
        pack_sequences=packing,
    )
    samples = []
    expanded_total = 0
    pad_multiple = _pack_pad_multiple()
    for i, ex in enumerate(prepared):
        sample_key = (
            sample_keys[i]
            if sample_keys is not None and i < len(sample_keys)
            else _prepared_sample_key(ex, batch_index=i)
        )

        chatml = prepared_to_chatml_sample(ex, key=sample_key)
        encoded = encoder.encode_sample(chatml)
        if capacity_planned:
            text_length = int(encoded.input_ids.size(0))
            aligned_text_length = text_length + (-text_length) % pad_multiple
            image_tiles = encoded.num_image_tiles
            visual_groups = int(image_tiles.numel()) if image_tiles is not None else 0
            visual_tokens = int(image_tiles.sum().item()) if image_tiles is not None else 0
            expanded_length = aligned_text_length - visual_groups + visual_tokens
            candidate_total = expanded_total + expanded_length
            physical_total = candidate_total + (-candidate_total) % pad_multiple
            if physical_total > seq_length:
                logger.warning(
                    "Skipping packed sample %s: expanded pack would be %d tokens (limit=%d)",
                    sample_key,
                    physical_total,
                    seq_length,
                )
                continue
            expanded_total = candidate_total
        log_encoded_sample(
            encoded,
            encoder._tokenizer,
            sample_key=sample_key,
            encoder_name=type(encoder).__name__,
            split=processed_prompt_split,
        )
        samples.append(encoded)
    if not samples:
        raise ValueError(f"No sample in the planned pack fits seq_length={seq_length}")
    batch = encoder.batch(samples)
    out = encoder.encode_batch(batch)
    tokens = out.pop("tokens", None)
    if tokens is not None:
        out["input_ids"] = tokens
    if batch.cu_seqlens is not None:
        out["cu_seqlens"] = batch.cu_seqlens
        out["cu_seqlens_unpadded"] = batch.cu_seqlens_unpadded
        out["cu_seqlens_argmin"] = batch.cu_seqlens_argmin
        out["cu_seqlens_unpadded_argmin"] = batch.cu_seqlens_argmin
        out["max_seqlen"] = batch.max_seqlen
    if packing and out.get("cu_seqlens") is not None:
        out = _align_packed_segments_for_parallel(out, pad_multiple=_pack_pad_multiple())
        global _pack_log_count
        if _pack_log_count < 3:
            _pack_log_count += 1
            cu = out["cu_seqlens"].tolist()
            segment_lengths = [int(cu[i + 1] - cu[i]) for i in range(len(cu) - 1)]
            logger.info(
                "packed pre-expansion: n_packed=%d input_tokens=%d input_lengths=%s",
                len(segment_lengths),
                int(cu[-1]),
                segment_lengths,
            )
    return out


def conversation_jsonl_collate_fn(
    examples: list,
    processor,
    *,
    seq_length: int = 4096,
    pack_sequences: bool = False,
    max_video_frames: int = 128,
    video_sample_fps: float = 2.0,
    processed_prompt_split: str = "train",
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("conversation_jsonl_collate_fn received an empty batch")
    capacity_planned = len(examples) == 1 and isinstance(examples[0], list)
    if capacity_planned:
        examples = examples[0]
        if not examples:
            raise ValueError("conversation_jsonl_collate_fn received an empty planned pack")

    sample_keys = [_prepared_sample_key(example, batch_index=i) for i, example in enumerate(examples)]
    prepared = [
        prepare_collate_example(
            example,
            max_video_frames=max_video_frames,
            video_sample_fps=video_sample_fps,
        )
        for example in examples
    ]
    return collate_prepared_via_task_encoder(
        prepared,
        processor,
        seq_length=seq_length,
        pack_sequences=pack_sequences,
        capacity_planned=capacity_planned,
        sample_keys=sample_keys,
        processed_prompt_split=processed_prompt_split,
    )


# --- LLaVA forward hooks (packed [1, L] microbatches only) ---

_hooks_installed = False
_llava_pack_log_count = 0


def _text_cu_from_packed(packed: Any) -> torch.Tensor | None:
    text_cu = getattr(packed, "cu_seqlens_q_padded", None)
    return text_cu if text_cu is not None else getattr(packed, "cu_seqlens_q", None)


def _is_packed_microbatch(input_ids: torch.Tensor | None, text_cu: torch.Tensor | None) -> bool:
    return (
        input_ids is not None
        and text_cu is not None
        and input_ids.dim() == 2
        and input_ids.size(0) == 1
        and text_cu.numel() > 1
    )


def _extend_cu_tail(cu: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    cu_list = [int(x) for x in cu.reshape(-1).tolist()]
    if not cu_list or seq_len <= cu_list[-1]:
        return cu
    cu_list[-1] = seq_len
    return torch.tensor(cu_list, dtype=cu.dtype, device=cu.device)


def _combined_cu_from_preprocess_positions(
    input_ids: torch.Tensor,
    text_cu: torch.Tensor,
    num_image_tiles: torch.Tensor,
    *,
    image_token_index: int,
    img_seq_len: int = 1,
) -> torch.Tensor:
    """Mirror LLaVA _preprocess_data new_position_ids segment boundaries."""
    mask = input_ids == image_token_index
    tiles = num_image_tiles.reshape(-1).to(device=input_ids.device, dtype=torch.int)
    n_img = int(mask.sum().item())
    if tiles.numel() != n_img:
        if tiles.numel() > n_img:
            tiles = tiles[:n_img]
        else:
            pad_val = int(tiles[-1].item()) if tiles.numel() else 1
            tiles = torch.nn.functional.pad(tiles, (0, n_img - tiles.numel()), value=pad_val)

    mask_lens = mask.int().clone()
    mask_lens[mask] = tiles * img_seq_len - 1
    new_pos = torch.cumsum(mask_lens + 1, dim=-1) - 1

    cu_text = [int(x) for x in text_cu.reshape(-1).tolist()]
    if len(cu_text) < 2:
        total = int(new_pos.reshape(-1)[-1].item()) + 1
        return torch.tensor([0, total], dtype=torch.int32, device=input_ids.device)

    cu_out = [0]
    for boundary in cu_text[1:-1]:
        if boundary >= input_ids.size(1):
            break
        cu_out.append(int(new_pos.reshape(-1)[boundary].item()))
    cu_out.append(int(new_pos.reshape(-1)[-1].item()) + 1)
    return torch.tensor(cu_out, dtype=torch.int32, device=input_ids.device)


def _packed_seq_params_from_cu(combined_cu: torch.Tensor, *, total_tokens: int) -> Any:
    seg_lens = [int(combined_cu[i + 1] - combined_cu[i]) for i in range(combined_cu.numel() - 1)]
    max_seqlen = torch.tensor(max(seg_lens) if seg_lens else 0, dtype=torch.int32, device=combined_cu.device)
    argmin = torch.tensor(combined_cu.numel(), dtype=torch.int32, device="cpu")
    return get_packed_seq_params(
        {
            "cu_seqlens": combined_cu,
            "cu_seqlens_unpadded": combined_cu.clone(),
            "cu_seqlens_argmin": argmin,
            "cu_seqlens_unpadded_argmin": argmin.clone(),
            "max_seqlen": max_seqlen,
            "total_tokens": total_tokens,
        }
    )


def _combined_seq_len(final_embedding: torch.Tensor) -> int:
    return int(final_embedding.shape[0] if final_embedding.dim() == 3 else final_embedding.shape[1])


def _parallel_shard_factor(model) -> int:  # type: ignore[no-untyped-def]
    factor = 1
    if getattr(model, "pre_process", True):
        if getattr(model, "sequence_parallel_lm", False):
            factor = max(factor, int(getattr(model, "tensor_model_parallel_size_lm", 1)))
    return max(1, factor)


def _install_llava_packed_hooks() -> None:
    from megatron.core.models.multimodal.llava_model import LLaVAModel

    _orig_forward = LLaVAModel.forward
    _orig_preprocess = LLaVAModel._preprocess_data
    _orig_parallel = LLaVAModel._process_embedding_token_parallel

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self._avlm_pack_ctx = None
        input_ids = kwargs.get("input_ids", args[1] if len(args) > 1 else None)
        packed = kwargs.get("packed_seq_params")
        num_image_tiles = kwargs.get("num_image_tiles")
        text_cu = _text_cu_from_packed(packed) if packed is not None else None
        if _is_packed_microbatch(input_ids, text_cu) and num_image_tiles is not None:
            tiles = (
                num_image_tiles
                if torch.is_tensor(num_image_tiles)
                else torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)
            )
            self._avlm_pack_ctx = (text_cu.detach(), tiles.detach(), packed)
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            self._avlm_pack_ctx = None

    def _preprocess_data(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        ctx = getattr(self, "_avlm_pack_ctx", None)
        input_ids = args[2] if len(args) > 2 else kwargs.get("input_ids")
        num_image_tiles = args[8] if len(args) > 8 else kwargs.get("num_image_tiles")

        combined_cu = None
        text_cu = None
        if ctx is not None and _is_packed_microbatch(input_ids, ctx[0]) and num_image_tiles is not None:
            text_cu, _, _ = ctx
            img_seq_len = 1 if getattr(self, "dynamic_resolution", False) else int(getattr(self, "img_seq_len", 1))
            combined_cu = _combined_cu_from_preprocess_positions(
                input_ids,
                text_cu,
                num_image_tiles,
                image_token_index=int(self.image_token_index),
                img_seq_len=img_seq_len,
            )

        final_embedding, final_labels, final_loss_mask = _orig_preprocess(self, *args, **kwargs)

        if combined_cu is not None and final_embedding is not None:
            actual = _combined_seq_len(final_embedding)
            rebuilt = int(combined_cu.reshape(-1)[-1].item())
            if rebuilt != actual:
                combined_cu = _extend_cu_tail(combined_cu, seq_len=actual)
                logger.warning("packed cu tail sync: %d -> %d", rebuilt, actual)
            self._avlm_combined_packed_cu = combined_cu
            if self.context_parallel_lm == 1 and not self.sequence_parallel_lm:
                rebuilt_params = _packed_seq_params_from_cu(combined_cu, total_tokens=actual)
                for field in dataclasses.fields(rebuilt_params):
                    setattr(ctx[2], field.name, getattr(rebuilt_params, field.name))
                self._avlm_combined_packed_cu = None
            global _llava_pack_log_count
            if _llava_pack_log_count < 5:
                _llava_pack_log_count += 1
                expanded_boundaries = [int(x) for x in combined_cu.tolist()]
                expanded_lengths = [
                    expanded_boundaries[i + 1] - expanded_boundaries[i]
                    for i in range(len(expanded_boundaries) - 1)
                ]
                expanded_total = expanded_boundaries[-1]
                hard_limit = os.environ.get("SEQ_LENGTH", "").strip()
                expanded_usage = f"{expanded_total}/{hard_limit}" if hard_limit else str(expanded_total)
                logger.info(
                    "packed multimodal: n_packed=%d expanded_tokens=%s expanded_lengths=%s",
                    len(expanded_lengths),
                    expanded_usage,
                    expanded_lengths,
                )

        return final_embedding, final_labels, final_loss_mask

    def _process_embedding_token_parallel(self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params):  # type: ignore[no-untyped-def]
        combined_cu = getattr(self, "_avlm_combined_packed_cu", None)
        if combined_cu is not None and packed_seq_params is not None:
            seq_len = _combined_seq_len(combined_embeddings)
            shard_factor = _parallel_shard_factor(self)
            remainder = seq_len % shard_factor
            if remainder:
                combined_cu = _extend_cu_tail(combined_cu, seq_len=seq_len + (shard_factor - remainder))
            packed_seq_params = _packed_seq_params_from_cu(
                combined_cu,
                total_tokens=int(combined_cu[-1].item()),
            )
            self._avlm_combined_packed_cu = None

        return _orig_parallel(self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params)

    LLaVAModel.forward = forward  # type: ignore[method-assign]
    LLaVAModel._preprocess_data = _preprocess_data  # type: ignore[method-assign]
    LLaVAModel._process_embedding_token_parallel = _process_embedding_token_parallel  # type: ignore[method-assign]


def install_packed_training_hooks() -> None:
    """Idempotent. Unpacked batches are no-ops inside the hooks."""
    global _hooks_installed
    if _hooks_installed:
        return
    _install_llava_packed_hooks()
    _hooks_installed = True
    logger.info("AVLM packed-training hooks installed")
