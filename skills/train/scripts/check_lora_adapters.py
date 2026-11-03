#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that every LoRA adapter in a checkpoint was actually trained.

LoRA initializes each ``lora_B`` matrix to zero (AutoModel and PEFT defaults). An adapter whose module never
receives a gradient (for example, a projection that a fused kernel calls by its
raw weight, bypassing the LoRA wrapper) keeps ``lora_B`` exactly zero, while
the run still logs finite losses and reaches its last step. This script reads
the adapter ``*.safetensors`` files under a checkpoint directory, groups the
``lora_B`` tensors by module (layer numbers replaced by ``*``), and reports
groups whose ``lora_B`` is all zero.

Expected targets: ``--expect`` names must each match a group with LoRA tensors
(``expect_missing`` otherwise). The ``target_modules`` list saved next to the
adapter (``adapter_config.json`` or ``automodel_peft_config.json``) is compared
too, but only as a note (``config_targets_not_in_checkpoint``): AutoModel can list
modules there that the recipe excluded, so it never fails the check by itself.

Standard library only: runs on a login node without torch.

Exit status: 0 = every adapter group was updated, 1 = at least one ``lora_B``
is all zero (inactive) or an ``--expect`` name has no LoRA tensors, 2 = no LoRA tensors in safetensors format found (for
example a Megatron-Bridge checkpoint) or a usage error.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import struct
import sys
from pathlib import Path

CHUNK = 64 << 20  # bytes read at a time


def _lora_split(key: str) -> tuple[str, str] | None:
    """Return (module path, 'A'|'B') for a LoRA tensor key, else None."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        low = part.lower()
        if low.startswith("lora") and low[-1:] in ("a", "b"):
            return ".".join(parts[:i]), low[-1].upper()
    return None


def _group(module: str) -> str:
    """Module path with layer and expert numbers replaced by '*'."""
    path = re.sub(r"^base_model\.model\.", "", module)
    return ".".join("*" if p.isdigit() else p for p in path.split("."))


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def _all_zero(path: Path, start: int, end: int) -> bool:
    """True if the byte range holds only zeros (+0.0 in any float dtype)."""
    with path.open("rb") as f:
        f.seek(start)
        left = end - start
        while left:
            buf = f.read(min(CHUNK, left))
            if not buf:
                break
            if buf.count(0) != len(buf):
                return False
            left -= len(buf)
    return True


CONFIG_NAMES = ("adapter_config.json", "automodel_peft_config.json")


def _saved_targets(files: list[Path]) -> list[str]:
    """target_modules lists saved next to the adapter files (regex strings are skipped)."""
    targets: set[str] = set()
    for folder in {f.parent for f in files}:
        for name in CONFIG_NAMES:
            cfg = folder / name
            if not cfg.is_file():
                continue
            try:
                data = json.loads(cfg.read_text())
            except (OSError, ValueError):
                continue
            stack = [data]
            while stack:  # target_modules may sit in a nested peft section
                obj = stack.pop()
                if isinstance(obj, dict):
                    tm = obj.get("target_modules")
                    if isinstance(tm, list):
                        targets.update(_group(str(x)) for x in tm)
                    stack.extend(v for v in obj.values() if isinstance(v, (dict, list)))
                elif isinstance(obj, list):
                    stack.extend(v for v in obj if isinstance(v, (dict, list)))
    return sorted(targets)


def _matches(group: str, target: str) -> bool:
    return fnmatch.fnmatch(group, target) or fnmatch.fnmatch(group, "*" + target) or group.endswith("." + target)


def check(root: Path, expect: list[str] | None = None) -> dict:
    root = root.resolve()  # follow a LATEST link
    if root.is_file():
        files = [root]
    else:
        files = sorted(Path(d) / n for d, _, names in os.walk(root, followlinks=True)
                       for n in names if n.endswith(".safetensors"))
    groups: dict[str, dict] = {}
    a_tensors = 0
    for path in files:
        header, offset = _read_header(path)
        for key, info in header.items():
            split = _lora_split(key)
            if split is None:
                continue
            module, which = split
            if which == "A":
                a_tensors += 1
                continue
            g = groups.setdefault(_group(module), {"lora_B": 0, "all_zero": 0, "example": key})
            g["lora_B"] += 1
            start, end = info["data_offsets"]
            if _all_zero(path, offset + start, offset + end):
                g["all_zero"] += 1

    inactive = sorted(k for k, g in groups.items() if g["all_zero"] == g["lora_B"])
    expect_missing = [e for e in (expect or []) if not any(_matches(k, _group(e)) for k in groups)]
    cfg_missing = [e for e in _saved_targets(files) if not any(_matches(k, e) for k in groups)]
    partial = sorted(k for k, g in groups.items() if 0 < g["all_zero"] < g["lora_B"])
    if not groups:
        status = "Not checked: no LoRA tensors in safetensors files"
    elif (inactive or partial) and expect_missing:
        status = "Inactive and missing adapters"
    elif inactive or partial:
        status = "Inactive adapters"
    elif expect_missing:
        status = "Expected adapters missing"
    else:
        status = "All adapters updated"
    return {
        "Adapter status": status,
        "checkpoint": str(root),
        "safetensors_files": len(files),
        "lora_A_tensors": a_tensors,
        "lora_B_tensors": sum(g["lora_B"] for g in groups.values()),
        "inactive_groups": inactive,
        "partly_inactive_groups": {k: f"{groups[k]['all_zero']}/{groups[k]['lora_B']} zero" for k in partial},
        "expect_missing": expect_missing,
        "config_targets_not_in_checkpoint": cfg_missing,
        "groups": {k: {"lora_B": g["lora_B"], "all_zero": g["all_zero"]} for k, g in sorted(groups.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path,
                        help="Checkpoint step directory (searched recursively), or one adapter .safetensors file")
    parser.add_argument("--expect", default="",
                        help="Comma-separated module names that must have LoRA tensors, e.g. in_proj,q_proj")
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        print(f"error: {args.checkpoint} does not exist", file=sys.stderr)
        return 2
    result = check(args.checkpoint, [e.strip() for e in args.expect.split(",") if e.strip()])
    print(json.dumps(result))
    if result["lora_B_tensors"] == 0:
        return 2
    bad = result["inactive_groups"] or result["partly_inactive_groups"] or result["expect_missing"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
