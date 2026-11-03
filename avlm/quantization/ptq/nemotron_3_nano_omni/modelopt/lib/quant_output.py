# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job output directory for quantize_omni.py."""

import os
import time
from pathlib import Path

_prefix = "nanov3omni"
_timestamp: int | None = None


def reset_output_path() -> None:
    global _timestamp
    _timestamp = None


def output_path() -> Path:
    global _timestamp
    if _timestamp is None:
        _timestamp = int(time.time())
    root = Path(os.environ.get("OMNI_QUANT_OUT_ROOT", "/tmp"))
    path = root / f"{_prefix}+{_timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path
