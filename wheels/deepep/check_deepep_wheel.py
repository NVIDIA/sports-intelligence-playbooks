#!/usr/bin/env python3
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

"""Exit 0 if deep_ep matches the current GPU; else 1 with install instructions.

Pip reports ``Version: 1.2.1+7febc6e`` for both pre- and post-Hopper wheels (the
profile tag is only in the ``.whl`` filename). We detect the build from ``-arch sm_*``
strings embedded in the installed extension.

Training preflight (only when YAML uses ``dispatcher: deepep``)::

    CONFIG_YAML=/path/to.yaml python3 check_deepep_wheel.py --pretrain

Set ``SKIP_DEEPEP_WHEEL_CHECK=1`` to skip. Remove the call from ``train_*.sh`` to drop
the gate entirely.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

_BAR = "=" * 78
_AUTOMODEL_PYTHON = Path("/opt/venv/bin/python3")
_ARCH_RE = re.compile(r"-arch sm_(\d+)")
_DISPATCHER_DEEPEP_RE = re.compile(r"dispatcher:\s*deepep")
_DISPATCHER_VALUE_RE = re.compile(r"dispatcher:\s*(\S+)")
_PRE_HOPPER = frozenset({80, 86, 89})
_POST_HOPPER = frozenset({90, 100, 103, 110, 120, 121})
_DISPATCHER_ENV_KEYS = ("DISPATCHER", "MODEL_DISPATCHER")


def resolve_training_dispatcher(config: str | None = None) -> str | None:
    """Effective MoE dispatcher: env override (DISPATCHER / MODEL_DISPATCHER) then recipe YAML."""
    for key in _DISPATCHER_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    path = (config or os.environ.get("CONFIG_YAML", "")).strip()
    if not path:
        return None
    yaml_path = Path(path)
    if not yaml_path.is_file():
        return None
    match = _DISPATCHER_VALUE_RE.search(yaml_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def training_uses_deepep(config: str | None = None) -> bool:
    return resolve_training_dispatcher(config) == "deepep"


def _emit_failure(summary: str, details: list[str]) -> int:
    lines = [
        "",
        _BAR,
        "DeepEP check failed — training will not start (dispatcher: deepep)",
        _BAR,
        "",
        f"  {summary}",
        "",
    ]
    if details:
        lines.append("  Details:")
        for item in details:
            lines.append(f"    • {item}")
        lines.append("")

    verify_py = _AUTOMODEL_PYTHON if _AUTOMODEL_PYTHON.is_file() else Path(sys.executable)
    lines.extend(
        [
            "  What to do (once per container session):",
            "",
            "    1. Install the pre-Hopper wheel (A100 / L40):",
            "         DEEPEP_WHEEL_PROFILE=pre-hopper-sm80-sm89 \\",
            "           bash wheels/deepep/install_deepep_wheel.sh",
            "",
            "    2. Verify:",
            f"         {verify_py} wheels/deepep/check_deepep_wheel.py",
            "",
            "    3. Re-run: bash avlm/training/automodel/lora/slurm/interactive/train_interactive.sh",
            "",
            "  Notes:",
            "    • H100+: DEEPEP_WHEEL_PROFILE=post-hopper-sm90-sm100-sm120",
            "    • No DeepEP: dispatcher: torch in your training YAML",
            "    • Skip gate: SKIP_DEEPEP_WHEEL_CHECK=1 or remove --pretrain from train_*.sh",
            "    • wheels/deepep/DEEPEP.MD",
            "",
            _BAR,
            "",
        ]
    )
    print("\n".join(lines), file=sys.stderr)
    return 1


def _gpu_info() -> tuple[int | None, str]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
            import torch
    except ImportError:
        return None, "CUDA unavailable (torch not found)"
    if not torch.cuda.is_available():
        return None, "CUDA unavailable"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    return major, f"{name} — SM {major}.{minor}"


def _extension_libs(ep_dir: Path) -> list[Path]:
    """Return native libs for deep_ep (wheel installs .so next to the package, not inside it)."""
    site = ep_dir.parent
    libs = sorted(site.glob("deep_ep*.so")) + sorted(site.glob("hybrid_ep*.so"))
    if not libs:
        libs = sorted(ep_dir.glob("*.so"))
    return libs


def _primary_extension(ep_dir: Path) -> Path | None:
    libs = _extension_libs(ep_dir)
    for lib in libs:
        if "deep_ep_cpp" in lib.name:
            return lib
    return libs[0] if libs else None


def _compiled_archs(so_path: Path) -> set[int]:
    try:
        text = subprocess.check_output(["strings", str(so_path)], text=True, errors="ignore")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    return {int(m.group(1)) for m in _ARCH_RE.finditer(text)}


def _archs_on_python(py: Path) -> set[int] | None:
    script = """
from pathlib import Path
import deep_ep
p = Path(deep_ep.__file__).resolve().parent
site = p.parent
libs = sorted(site.glob('deep_ep*.so')) + sorted(site.glob('hybrid_ep*.so'))
print(libs[0] if libs else '')
"""
    try:
        so = subprocess.check_output([str(py), "-c", script], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not so:
        return None
    return _compiled_archs(Path(so))


def _format_archs(archs: set[int]) -> str:
    if not archs:
        return "(could not read -arch sm_* from extension; is strings installed?)"
    parts = []
    if archs & _PRE_HOPPER:
        parts.append("pre-Hopper: " + ", ".join(f"sm_{a}" for a in sorted(archs & _PRE_HOPPER)))
    if archs & _POST_HOPPER:
        parts.append("Hopper+: " + ", ".join(f"sm_{a}" for a in sorted(archs & _POST_HOPPER)))
    other = archs - _PRE_HOPPER - _POST_HOPPER
    if other:
        parts.append("other: " + ", ".join(f"sm_{a}" for a in sorted(other)))
    return "; ".join(parts) if parts else f"sm_{min(archs)}..sm_{max(archs)}"


def run_check() -> int:
    try:
        import deep_ep
    except ImportError:
        return _emit_failure(
            "deep_ep is not installed for this Python.",
            [f"Python: {sys.executable}"],
        )

    from importlib.metadata import version

    ver = version("deep_ep")
    ep_dir = Path(deep_ep.__file__).resolve().parent
    so_path = _primary_extension(ep_dir)
    archs = _compiled_archs(so_path) if so_path else set()

    major, gpu_line = _gpu_info()

    if major is None:
        print(f"\n  deep_ep OK (no CUDA)\n  version: {ver}\n  path: {ep_dir}\n")
        return 0

    has_pre = bool(archs & _PRE_HOPPER)
    has_post = bool(archs & _POST_HOPPER)

    if major < 9 and not has_pre:
        details = [
            f"Python: {sys.executable}",
            gpu_line,
            f"Package version (pip): {ver}",
            f"Extension: {so_path or '(missing deep_ep*.so next to package)'}",
            f"Kernels: {_format_archs(archs)}",
            "Need -arch sm_80 (or sm_86/sm_89) from the pre-Hopper lustre wheel.",
        ]
        if _AUTOMODEL_PYTHON.is_file() and Path(sys.executable).resolve() != _AUTOMODEL_PYTHON.resolve():
            other = _archs_on_python(_AUTOMODEL_PYTHON)
            if other and (other & _PRE_HOPPER):
                details.append(f"{_AUTOMODEL_PYTHON} has pre-Hopper kernels — use that Python for training.")
        return _emit_failure("Installed deep_ep cannot run on this GPU (missing Ampere kernels).", details)

    if major >= 9 and has_pre and not has_post:
        return _emit_failure(
            "Installed deep_ep is pre-Hopper-only.",
            [
                f"Python: {sys.executable}",
                gpu_line,
                f"Kernels: {_format_archs(archs)}",
                "Install the post-Hopper wheel for H100+.",
            ],
        )

    print(
        textwrap.dedent(
            f"""
            {_BAR}
              deep_ep OK
            {_BAR}
              python:  {sys.executable}
              GPU:     {gpu_line}
              version: {ver}
              kernels: {_format_archs(archs)}
              path:    {ep_dir}
            {_BAR}
            """
        ).strip()
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify deep_ep wheel matches this GPU.")
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run only if training uses dispatcher: deepep (for train_*.sh preflight).",
    )
    parser.add_argument(
        "--needs-deepep",
        action="store_true",
        help="Exit 0 if DeepEP install is needed, 1 if not (used by install_deepep_wheel.sh).",
    )
    args = parser.parse_args(argv)

    config = os.environ.get("CONFIG_YAML", "").strip()

    if args.needs_deepep:
        if os.environ.get("SKIP_DEEPEP_INSTALL", "0") == "1" or os.environ.get(
            "SKIP_DEEPEP_INSTALL_ON_BATCH", "0"
        ) == "1":
            return 1
        if not config:
            print("error: CONFIG_YAML must be set for --needs-deepep", file=sys.stderr)
            return 2
        if not Path(config).is_file():
            print(f"error: CONFIG_YAML is not a file: {config}", file=sys.stderr)
            return 2
        return 0 if training_uses_deepep(config) else 1

    if args.pretrain:
        if os.environ.get("SKIP_DEEPEP_WHEEL_CHECK", "0") == "1":
            return 0
        if not config:
            print("error: CONFIG_YAML must be set for --pretrain", file=sys.stderr)
            return 1
        if not Path(config).is_file():
            print(f"error: CONFIG_YAML is not a file: {config}", file=sys.stderr)
            return 1
        if not training_uses_deepep(config):
            return 0

    return run_check()


if __name__ == "__main__":
    raise SystemExit(main())
