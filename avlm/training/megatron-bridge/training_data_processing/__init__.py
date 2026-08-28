# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training data processing for JSONL conversation SFT (load, collate, Bridge recipe).

On-disk paths and format are configured in the training YAML
(``path_or_dataset``, ``jsonl_format``, ``video_root``).

Deploy runs at train time when ``run.recipe`` (from the training YAML) names a function
defined in ``training_data_processing/recipe.py``.
"""

from .collate import conversation_jsonl_collate_fn
from .dataset import JsonlFormat, load_jsonl_examples, normalize_jsonl_format
from .recipe import ConversationJsonlProvider, conversation_jsonl_sft_config

__all__ = [
    "ConversationJsonlProvider",
    "JsonlFormat",
    "conversation_jsonl_collate_fn",
    "conversation_jsonl_sft_config",
    "load_jsonl_examples",
    "normalize_jsonl_format",
]
