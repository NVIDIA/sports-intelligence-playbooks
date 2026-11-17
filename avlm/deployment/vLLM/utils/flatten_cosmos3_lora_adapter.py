# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a Cosmos3 PEFT LoRA adapter so vLLM applies it.

PEFT names the adapted modules after the transformers model tree:
`base_model.model.model.language_model.layers.<n>.<...>.lora_{A,B}.weight`. vLLM's
Cosmos3 implementation maps the checkpoint's flat layout instead, with the decoder at
`layers.<n>.<...>`, and its LoRA loader resolves adapter keys against those module names.
Keys carrying the transformers prefix match nothing, vLLM loads the adapter without an
error, and requests routed to it run the base model unchanged.

This script rewrites the prefix to `base_model.model.layers.<n>.<...>` and copies
`adapter_config.json`, producing an adapter directory for `--lora-modules`.
Keys outside the language model (vision tower, projector) are rejected: the adapter
must be language-model-only, as the Cosmos3 LoRA recipe trains it.

The flat layout is the one vLLM uses for nvidia/Cosmos3-Nano
(Cosmos3OmniForConditionalGeneration in transformers). Adapters of other base classes
are refused: Cosmos3-Edge has a different decoder layout and its vLLM model class does not
implement LoRA.

usage:
  python avlm/deployment/vLLM/utils/flatten_cosmos3_lora_adapter.py \
    --adapter /path/to/adapter --out /path/to/adapter_flat
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

SUPPORTED_BASE_CLASS = "Cosmos3OmniForConditionalGeneration"
PEFT_PREFIX = "base_model.model.model.language_model."
VLLM_PREFIX = "base_model.model."
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def flatten_key(key: str) -> str:
    if not key.startswith(PEFT_PREFIX):
        raise SystemExit(f"not a language-model adapter key: {key}")
    flat = VLLM_PREFIX + key[len(PEFT_PREFIX):]
    if not flat.startswith(VLLM_PREFIX + "layers."):
        raise SystemExit(f"unexpected module under the language model: {key}")
    return flat


def check_base_class(adapter: str) -> None:
    cfg = json.load(open(os.path.join(adapter, ADAPTER_CONFIG)))
    base_class = (cfg.get("auto_mapping") or {}).get("base_model_class", "")
    if base_class != SUPPORTED_BASE_CLASS:
        raise SystemExit(f"adapter base class {base_class!r} is not {SUPPORTED_BASE_CLASS}; the flat layout only matches Cosmos3-Nano")


def flatten(adapter: str, out: str) -> int:
    check_base_class(adapter)
    os.makedirs(out, exist_ok=True)
    tensors = load_file(os.path.join(adapter, ADAPTER_WEIGHTS))
    flat = {flatten_key(k): v for k, v in tensors.items()}
    save_file(flat, os.path.join(out, ADAPTER_WEIGHTS), metadata={"format": "pt"})
    shutil.copy(os.path.join(adapter, ADAPTER_CONFIG), os.path.join(out, ADAPTER_CONFIG))
    return len(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory (adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--out", required=True, help="output directory for the flattened adapter")
    args = parser.parse_args()
    count = flatten(args.adapter, args.out)
    print(f"wrote {count} tensors to {args.out}")


if __name__ == "__main__":
    main()
