# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for vLLM OpenAI api_server (see vllm_server.sh for flags)."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path


def _register_hub_checkpoint_video_loader() -> None:
    loader = Path(__file__).resolve().parent / "utils" / "hub_checkpoint_video_loader.py"
    spec = importlib.util.spec_from_file_location("vllm_deploy_hub_checkpoint_video_loader", loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {loader}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main() -> None:
    _register_hub_checkpoint_video_loader()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
