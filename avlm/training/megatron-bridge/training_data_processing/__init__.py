# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
