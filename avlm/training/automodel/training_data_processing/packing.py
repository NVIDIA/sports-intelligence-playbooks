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

"""Runtime patches so video-sound VLM data can use neat packing without editing nemo_automodel.

Imported by ``avlm.training.automodel.training_data_processing.training_recipe`` before training starts.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from avlm.training.automodel.training_data_processing.video_sound_utils import (
    video_path_from_content_item,
)

logger = logging.getLogger(__name__)

_PATCHED = False
_pack_video_max_frames: int | None = None
_pack_video_sample_fps: float | None = None

# Per-frame video token budget for the packing length estimator, scaled from the 135 measured at
# the model's default 1024 patches/frame. Rounded up: under-estimating overfills packs and drops
# samples as overlong, while over-estimating only under-fills them.
_REFERENCE_VIDEO_TARGET_NUM_PATCHES = 1024
_DEFAULT_TOKENS_PER_VIDEO_FRAME = 135
_pack_tokens_per_video_frame: int = _DEFAULT_TOKENS_PER_VIDEO_FRAME


def _tokens_per_video_frame(cfg_processor) -> int:
    """Token cost implied by ``processor.video_target_num_patches``: 512 -> 68, 1024 -> 135, 2048 -> 270."""
    target = cfg_processor.get("video_target_num_patches") if cfg_processor is not None else None
    if not target:
        return _DEFAULT_TOKENS_PER_VIDEO_FRAME
    return math.ceil(_DEFAULT_TOKENS_PER_VIDEO_FRAME * int(target) / _REFERENCE_VIDEO_TARGET_NUM_PATCHES)


@lru_cache(maxsize=None)
def _read_video_metadata(video_path: str) -> tuple[float, int, float]:
    """Read duration/frame count/FPS once per process."""
    import decord

    video_reader = decord.VideoReader(video_path, num_threads=1)
    try:
        total_frames = len(video_reader)
        fps = float(video_reader.get_avg_fps())
        duration = total_frames / max(fps, 1e-6)
    finally:
        del video_reader
    return duration, total_frames, fps


def apply_video_sound_packing_patches() -> None:
    """Patch AutoModel VLM finetune + packing once per process for video-sound data."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_build_dataloader()
    _patch_video_sound_packing()
    _patch_nemotron_sound_alignment()
    _patch_fused_linear_ce_for_vlm()
    _PATCHED = True
    logger.info("Applied avlm video-sound packing patches.")


def _patch_fused_linear_ce_for_vlm() -> None:
    """Let VLM wrappers use ``FusedLinearCrossEntropy`` for long packed sequences.

    ``FinetuneRecipeForVLM`` checks ``_supports_logits_to_keep`` on the top-level
    VLM wrapper, but some models forward ``logits_to_keep`` to ``language_model``
    via ``**kwargs``. Without this patch, setup downgrades to ``MaskedCrossEntropy``,
    materializing a full ``[B, S, vocab]`` logits tensor (~10 GiB for 20k tokens).
    """
    import nemo_automodel.components.utils.model_utils as model_utils
    import nemo_automodel.recipes.vlm.finetune as finetune_mod

    if getattr(model_utils._supports_logits_to_keep, "_avlm_vlm_fused_ce_patched", False):
        return

    _orig = model_utils._supports_logits_to_keep

    def _supports_logits_to_keep(model) -> bool:
        if _orig(model):
            return True
        language_model = getattr(model, "language_model", None)
        return language_model is not None and _orig(language_model)

    _supports_logits_to_keep._avlm_vlm_fused_ce_patched = True  # type: ignore[attr-defined]
    model_utils._supports_logits_to_keep = _supports_logits_to_keep
    finetune_mod._supports_logits_to_keep = _supports_logits_to_keep

# replaces legacy _split_masks_for_hybrid_attention
def _ensure_packed_seq_ids(result: dict[str, Any], batch: list[dict]) -> None:
    """Restore indexed IDs AutoModel omits for all-singleton 4D-mask batches."""

    # If more than one sample per back automodel should
    # already pass _packed_seq_ids
    if "_packed_seq_ids" in result:
        return

    attention_mask = result.get("attention_mask")
    input_ids = result.get("input_ids")

    # Cannot determine attention mask - from legacy implementation
    if attention_mask is None:
        return

    # No input ids return - from legacy implementation
    if input_ids is None:
        return

    # If not 4D then Flash attention, do not need
    if attention_mask.dim() != 4:
        return

    if input_ids.dim() != 2 or input_ids.shape[0] != len(batch):
        raise ValueError("Packed input_ids must have shape [batch, sequence]")

    # Preserve pre-padding sequence ownership; EOS and padding may share a token ID.
    indexed = torch.zeros_like(input_ids, dtype=torch.long)
    for row, item in enumerate(batch):
        sample_ids = torch.as_tensor(item["attention_mask"], dtype=torch.long, device=indexed.device)
        if sample_ids.dim() != 1 or sample_ids.numel() > indexed.shape[1]:
            raise ValueError("Packed sequence IDs must be 1D and fit the collated sequence length")
        indexed[row, : sample_ids.numel()] = sample_ids
    result["_packed_seq_ids"] = indexed

def _patch_nemotron_sound_alignment() -> None:
    """Remove per-clip audio padding before sound embeddings are flattened."""
    from nemo_automodel.components.models.nemotron_omni.model import (
        NemotronOmniForConditionalGeneration,
    )

    original = NemotronOmniForConditionalGeneration.extract_sound_feature
    if getattr(original, "_avlm_packed_sound_alignment", False):
        return

    def extract_sound_feature(self, input_features, attention_mask=None):
        sound_embeds = original(self, input_features, attention_mask)
        if sound_embeds.dim() != 3 or sound_embeds.shape[0] <= 1 or attention_mask is None:
            return sound_embeds

        # ParakeetFeatureExtractor's mask omits the final center-padded STFT frame.
        input_lengths = attention_mask.sum(dim=-1) + 1
        output_lengths = self.sound_encoder._get_subsampling_output_length(input_lengths).tolist()
        compact = torch.cat(
            [sound_embeds[i, : int(length)] for i, length in enumerate(output_lengths)],
            dim=0,
        )
        return compact.unsqueeze(0)

    extract_sound_feature._avlm_packed_sound_alignment = True  # type: ignore[attr-defined]
    NemotronOmniForConditionalGeneration.extract_sound_feature = extract_sound_feature

def _patch_build_dataloader() -> None:
    import nemo_automodel.recipes.vlm.finetune as finetune_mod

    if getattr(finetune_mod.build_dataloader, "_avlm_build_dataloader_patched", False):
        return

    _original = finetune_mod.build_dataloader

    def build_dataloader(*args: Any, **kwargs: Any):
        cfg_ds = kwargs.get("cfg_ds") if "cfg_ds" in kwargs else (args[0] if args else None)
        use_custom_pretokenize = False
        if cfg_ds is not None:
            use_custom_pretokenize = bool(
                cfg_ds.get(
                    "use_custom_pretokenize",
                    cfg_ds.get("custom_pretokenize_wrapper", cfg_ds.get("video_sound_pretokenize", False)),
                )
            )
        if use_custom_pretokenize:
            # 4th positional arg of build_dataloader; carries the recipe's ``processor:`` block.
            cfg_processor = (
                kwargs.get("cfg_processor") if "cfg_processor" in kwargs else (args[3] if len(args) > 3 else None)
            )
            return _build_dataloader_with_video_sound_pretokenize(_original, cfg_ds, cfg_processor, *args, **kwargs)
        return _original(*args, **kwargs)

    build_dataloader._avlm_build_dataloader_patched = True  # type: ignore[attr-defined]
    finetune_mod.build_dataloader = build_dataloader


def _build_dataloader_with_video_sound_pretokenize(original, cfg_ds, cfg_processor, *args: Any, **kwargs: Any):
    """Swap in :class:`VideoSoundPreTokenizedDatasetWrapper` for the pretokenize step only."""
    global _pack_video_max_frames, _pack_video_sample_fps, _pack_tokens_per_video_frame

    import nemo_automodel.components.datasets.vlm.datasets as vlm_datasets
    from avlm.training.automodel.training_data_processing.pretokenize import (
        DEFAULT_VIDEO_SAMPLE_FPS,
        VideoSoundPreTokenizedDatasetWrapper,
    )

    max_video_frames = int(cfg_ds.get("max_video_frames", 8))
    video_sample_fps = cfg_ds.get("video_sample_fps", DEFAULT_VIDEO_SAMPLE_FPS)
    if video_sample_fps is not None:
        video_sample_fps = float(video_sample_fps)
    saved_cls = vlm_datasets.PreTokenizedDatasetWrapper
    _pack_video_max_frames = max_video_frames
    _pack_video_sample_fps = video_sample_fps
    _pack_tokens_per_video_frame = _tokens_per_video_frame(cfg_processor)

    class _VideoSoundPretokenizedWrapper(VideoSoundPreTokenizedDatasetWrapper):
        def __init__(
            self,
            dataset,
            processor,
            max_length=None,
            max_retries=10,
            truncate=False,
            post_tokenize_hook=None,
            **pretokenize_kwargs: Any,
        ):
            super().__init__(
                dataset,
                processor,
                max_length=max_length,
                max_retries=max_retries,
                truncate=truncate,
                post_tokenize_hook=post_tokenize_hook,
                max_video_frames=max_video_frames,
                video_sample_fps=video_sample_fps,
                **pretokenize_kwargs,
            )

    vlm_datasets.PreTokenizedDatasetWrapper = _VideoSoundPretokenizedWrapper
    try:
        return original(*args, **kwargs)
    finally:
        vlm_datasets.PreTokenizedDatasetWrapper = saved_cls
        _pack_video_max_frames = None
        _pack_video_sample_fps = None

def _patch_video_sound_packing() -> None:
    import nemo_automodel.components.datasets.vlm.collate_fns as collate_mod
    import nemo_automodel.components.datasets.vlm.neat_packing_vlm as packing_mod

    if getattr(packing_mod._shift_sample, "_avlm_sound_packing_patched", False):
        return

    _orig_shift = packing_mod._shift_sample
    _orig_build = packing_mod._build_packed_vlm_sample
    _orig_collate = collate_mod.neat_packed_vlm_collater
    _orig_est_video = packing_mod._estimate_video_tokens
    _orig_est_sample = packing_mod._estimate_sample_length

    # Vision tokens per frame come from the module-level ``_pack_tokens_per_video_frame``, read at
    # call time since this patch runs before build_dataloader sets it.
    # audio is budgeted separately at 256 per frame
    _SOUND_TOKEN_BUDGET = 256
    _VIDEO_TEMPORAL_PATCH_SIZE = 2

    def _video_path(example: dict) -> str | None:
        audio_video_path = example.get("audio_video_path")
        if isinstance(audio_video_path, str):
            return audio_video_path
        for message in example.get("conversation", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        video = video_path_from_content_item(item)
                        if video is not None:
                            return video
        return None

    # now estimate media tokens instead of assuming max
    # future TODO esimate audio tokens
    def _estimated_sampled_frames(
        video_path: str,
        metadata: Any,
        max_video_frames: int,
        video_sample_fps: float | None,
    ) -> int:
        from avlm.training.automodel.training_data_processing.pretokenize import (
            compute_video_frame_indices,
        )

        try:
            if metadata is None:
                _, total_frames, video_fps = _read_video_metadata(video_path)
            else:
                _, total_frames, video_fps = metadata
                total_frames = int(total_frames)
                video_fps = float(video_fps)
        except Exception as exc:
            raise RuntimeError(f"Could not read video metadata for {video_path}") from exc
        return len(
            compute_video_frame_indices(
                total_frames,
                video_fps,
                max_video_frames=max_video_frames,
                video_sample_fps=video_sample_fps,
            )
        )

    def _estimate_sample_length(
        example,
        image_cfg=None,
        video_cfg=None,
        return_media_tokens=False,
    ):
        video_path = _video_path(example)
        if _pack_video_max_frames is not None and video_path is not None:
            precomputed = example.get("_text_tokens")
            if precomputed is not None:
                text_tokens = int(precomputed)
            else:
                total_chars = 0
                for msg in example.get("conversation", []):
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                total_chars += len(item.get("text", ""))
                text_tokens = total_chars // 3
            sampled_frames = _estimated_sampled_frames(
                video_path,
                example.get("_video_metadata"),
                _pack_video_max_frames,
                _pack_video_sample_fps,
            )
            media_tokens = sampled_frames * _pack_tokens_per_video_frame
            if example.get("audio_video_path"):
                media_tokens += _SOUND_TOKEN_BUDGET
            total = text_tokens + media_tokens
            if return_media_tokens:
                return total, media_tokens
            return total
        return _orig_est_sample(
            example,
            image_cfg=image_cfg,
            video_cfg=video_cfg,
            return_media_tokens=return_media_tokens,
        )

    def _materialize_pack(self, pack_idx: int) -> dict:
        from avlm.training.automodel.training_data_processing.pretokenize import SampleTooLongError
        from avlm.utils.batch_memory import maybe_gc_after_pack, release_vlm_batch

        bin_indices = self.bins[pack_idx]
        shifted_samples: list[dict] = []

        for sample_idx in bin_indices:
            try:
                sample = self.inner[sample_idx]
            except SampleTooLongError as exc:
                logger.warning(
                    "Pack %d: skipping overlong sample %d (%d tokens > max_length %d)",
                    pack_idx,
                    exc.idx,
                    exc.seq_len,
                    exc.max_length,
                )
                continue

            if self.has_mrope and self.get_rope_index is not None:
                mrope_pos = packing_mod._compute_mrope_position_ids(sample, self.get_rope_index)
                if mrope_pos is not None:
                    sample["position_ids"] = mrope_pos

            shifted = _shift_sample(sample, has_mrope=self.has_mrope)
            seq_len = shifted["input_ids"].shape[0]

            if seq_len > self.pack_size:
                logger.warning(
                    "Pack %d: sample %d has %d tokens (> pack_size %d), skipping.",
                    pack_idx,
                    sample_idx,
                    seq_len,
                    self.pack_size,
                )
                continue

            shifted_samples.append(shifted)

        total = 0
        kept: list[dict] = []
        for sample in shifted_samples:
            seq_len = sample["input_ids"].shape[0]
            if total + seq_len <= self.pack_size:
                kept.append(sample)
                total += seq_len
            else:
                logger.warning(
                    "Pack %d: skipping overflow sample (%d tokens, %d/%d used; not truncated).",
                    pack_idx,
                    seq_len,
                    total,
                    self.pack_size,
                )

        if not kept:
            logger.warning(
                "Pack %d: no valid samples after materialization; returning a padding-only pack.",
                pack_idx,
            )
            kept = [
                {
                    "input_ids": torch.tensor([], dtype=torch.long),
                    "labels": torch.tensor([], dtype=torch.long),
                }
            ]

        packed = _build_packed_vlm_sample(
            kept,
            self.pack_size,
            self.padding_idx,
            has_mrope=self.has_mrope,
        )

        for sample in shifted_samples:
            release_vlm_batch(sample)
        shifted_samples.clear()
        maybe_gc_after_pack()
        return packed

    def _estimate_video_tokens(vid_meta, video_cfg):
        if _pack_video_max_frames is not None:
            video_cfg = dict(video_cfg)
            video_cfg["max_frames"] = _pack_video_max_frames
        return _orig_est_video(vid_meta, video_cfg)

    def robust_collate(self, collate_fn):
        """Re-sample packs from ``PackedDatasetWrapper``, not inner pretokenized samples."""
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        return make_robust_collate(self, collate_fn, self.max_retries)

    def _shift_sample(sample: dict, has_mrope: bool = False) -> dict:
        out = _orig_shift(sample, has_mrope=has_mrope)
        for key in ("sound_features", "sound_attention_mask"):
            if key in sample and sample[key] is not None:
                out[key] = sample[key]
        return out

    def _build_packed_vlm_sample(
        samples: list[dict],
        pack_size: int,
        padding_idx: int,
        has_mrope: bool = False,
    ) -> dict:
        aligned_samples: list[dict] = []
        for sample in samples:
            pixels = sample.get("pixel_values_videos")
            if pixels is not None and pixels.shape[0] > 0:
                # Preserve per-video tubelet boundaries before concatenation. The model
                # normally pads each standalone odd-length video by repeating its last frame.
                pad_frames = (-pixels.shape[0]) % _VIDEO_TEMPORAL_PATCH_SIZE
                if pad_frames:
                    sample = dict(sample)
                    padding = pixels[-1:].expand(pad_frames, *pixels.shape[1:])
                    sample["pixel_values_videos"] = torch.cat((pixels, padding), dim=0)
            aligned_samples.append(sample)

        packed = _orig_build(aligned_samples, pack_size, padding_idx, has_mrope=has_mrope)
        sound_features_list: list[torch.Tensor] = []
        sound_mask_list: list[torch.Tensor] = []
        for sample in aligned_samples:
            if "sound_features" in sample and sample["sound_features"] is not None:
                sf = sample["sound_features"]
                sf = sf[0] if sf.dim() == 3 else sf
                sound_features_list.append(sf)
                sm = sample.get("sound_attention_mask")
                if sm is None:
                    sm = torch.ones(sf.shape[0], dtype=torch.long, device=sf.device)
                elif sm.dim() == 2:
                    sm = sm[0]
                sound_mask_list.append(sm)
        if sound_features_list:
            packed["sound_features"] = pad_sequence(sound_features_list, batch_first=True)
            packed["sound_attention_mask"] = pad_sequence(sound_mask_list, batch_first=True)
        return packed

    def neat_packed_vlm_collater(
        batch: list[dict],
        padding_idx: int = 0,
        max_length: int | None = None,
        attn_implementation: str = "sdpa",
    ) -> dict:
        result = _orig_collate(
            batch,
            padding_idx=padding_idx,
            max_length=max_length,
            attn_implementation=attn_implementation,
        )
        _ensure_packed_seq_ids(result, batch)
        for key in ("sound_features", "sound_attention_mask"):
            tensors = [x[key] for x in batch if key in x and x[key] is not None]
            if tensors:
                result[key] = torch.cat(tensors, dim=0)
        return result

    _shift_sample._avlm_sound_packing_patched = True  # type: ignore[attr-defined]
    packing_mod._shift_sample = _shift_sample
    packing_mod._build_packed_vlm_sample = _build_packed_vlm_sample
    packing_mod._estimate_video_tokens = _estimate_video_tokens
    packing_mod._estimate_sample_length = _estimate_sample_length
    packing_mod.PackedDatasetWrapper.__getitem__ = _materialize_pack
    packing_mod.PackedDatasetWrapper.robust_collate = robust_collate
    collate_mod.neat_packed_vlm_collater = neat_packed_vlm_collater
