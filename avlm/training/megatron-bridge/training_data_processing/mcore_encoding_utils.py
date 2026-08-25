# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Loss-mask helpers for Nemotron MoE multimodal SFT."""

from __future__ import annotations

from typing import Any

import numpy as np

_ASSISTANT_TURN_HEADER = "<|im_start|>assistant\n"
_IM_END_TOKEN = "<|" + "im_end" + "|>"


def resolve_im_end_token_id(tokenizer: Any) -> int:
    """Return the turn-end token id (``eos_token`` / ``<|im_end|>`` for Nemotron)."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    token_id = tokenizer.convert_tokens_to_ids(_IM_END_TOKEN)
    if token_id is not None:
        return int(token_id)
    raise ValueError(f"could not resolve im_end token id from tokenizer {type(tokenizer)!r}")


def resolve_assistant_turn_header_ids(tokenizer: Any) -> np.ndarray:
    """Token ids for ``<|im_start|>assistant\\n`` (nemotron6-moe assistant-turn prefix)."""
    header_ids = tokenizer.encode(_ASSISTANT_TURN_HEADER, add_special_tokens=False)
    if not header_ids:
        raise ValueError(f"tokenizer produced empty ids for assistant header {_ASSISTANT_TURN_HEADER!r}")
    return np.asarray(header_ids, dtype=np.int64)


def resolve_nemotron_moe_loss_mask_ids(tokenizer: Any) -> tuple[np.ndarray, int]:
    """Return ``(assistant_header_ids, im_end_id)`` for nemotron6-moe loss masking."""
    return resolve_assistant_turn_header_ids(tokenizer), resolve_im_end_token_id(tokenizer)


def build_nemotron_moe_assistant_loss_mask(
    tokens: np.ndarray,
    *,
    assistant_header_ids: np.ndarray,
    im_end_id: int,
) -> np.ndarray:
    """Supervise assistant spans (Megatron-LM ``MultimodalTokenizer`` nemotron6-moe rules)."""
    tokens_arr = np.asarray(tokens, dtype=np.int64)
    loss_mask = np.zeros(len(tokens_arr), dtype=np.float32)
    if len(tokens_arr) < 3:
        return loss_mask

    end_positions = np.where(tokens_arr == im_end_id)[0]
    if len(end_positions) == 0:
        return loss_mask

    header = np.asarray(assistant_header_ids, dtype=np.int64).reshape(-1)
    if len(header) < 3:
        return loss_mask
    p0, p1, p2 = int(header[0]), int(header[1]), int(header[2])

    pattern_matches = np.where(
        (tokens_arr[:-2] == p0) & (tokens_arr[1:-1] == p1) & (tokens_arr[2:] == p2)
    )[0]
    for match_pos in pattern_matches:
        lb = int(match_pos + 1)
        if lb + 2 >= len(tokens_arr):
            continue
        valid_ends = end_positions[end_positions > lb]
        if len(valid_ends) == 0:
            continue
        ub = int(valid_ends[0])
        content_start = lb + 3
        if content_start > ub:
            continue
        loss_mask[content_start : ub + 1] = 1.0
    return loss_mask
