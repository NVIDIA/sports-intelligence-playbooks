#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Stage AVLM modules on an import overlay (Bridge cache checkout stays pristine)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RECIPE_MODULES = ("dataset", "video_utils", "mcore_encoding_utils", "collate", "recipe")
OVERLAY_RECIPE_DIR = Path("megatron/bridge/recipes/nemotron_omni")
LLAVA_MODEL_REL = Path("megatron/core/models/multimodal/llava_model.py")
# Never leave these in the overlay — they shadow Bridge package __init__.py exports.
_STALE_OVERLAY_INITS = (
    "megatron/__init__.py",
    "megatron/bridge/__init__.py",
    "megatron/bridge/recipes/__init__.py",
    "megatron/bridge/recipes/nemotron_omni/__init__.py",
    "megatron/bridge/training/__init__.py",
    "megatron/bridge/training/utils/__init__.py",
)
_STAGE = (("recipe", OVERLAY_RECIPE_DIR, RECIPE_MODULES),)

_LLAVA_SOURCE_REPLACEMENTS = (
    (
        "restore spatial pixel_shuffle for explicit h/w grids",
        """    if h is not None or w is not None:
        assert h is not None and w is not None, "h and w must both be provided"
        assert h * w == x.shape[1], f"h*w ({h}*{w}={h*w}) must equal patches ({x.shape[1]})"
        r = int(1 / scale_factor)
        n, patches, c = x.shape
        return x.reshape(n, patches // (r * r), c * r * r)
    h = w = int(x.shape[1] ** 0.5)  # sq
    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, sq, sq, h_vision]""",
        """    if h is not None or w is not None:
        assert h is not None and w is not None, "h and w must both be provided"
    else:
        h = w = int(x.shape[1] ** 0.5)  # sq
    assert h * w == x.shape[1], f"h*w ({h}*{w}={h*w}) must equal patches ({x.shape[1]})"
    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, h, w, h_vision]""",
    ),
    (
        "use dynamic-resolution image token counts in preprocessing",
        """        # Packed dynamic-res path: each image's token count is in num_image_tiles (one entry
        # per image) and there is exactly 1 embedding per "tile", so img_seq_len collapses to 1.
        img_seq_len = 1 if is_packed_dynamic_res else self.img_seq_len""",
        """        # Dynamic-resolution inputs pass the real per-image token counts in num_image_tiles.
        # Static-tile inputs keep the fixed per-tile image sequence length.
        if getattr(self, "dynamic_resolution", False):
            img_seq_len = 1
        else:
            img_seq_len = self.img_seq_len""",
    ),
    (
        "use temporal post-shuffle token counts for dynamic-resolution video",
        """                # After temporal grouping each tubelet is one "tile" for LLaVAModel's
                # _preprocess_data; one entry per post-grouping image (images have 1 frame,
                # videos contribute ceil(nf/T) tubelets).
                num_image_tiles = torch.ones(
                    image_embeddings.shape[0], dtype=torch.int, device=image_embeddings.device
                )""",
        """                # After temporal grouping, each tubelet contributes its actual post-shuffle
                # token count. Dynamic-resolution preprocessing uses these counts directly.
                tokens_per_tubelet = (
                    image_embeddings.shape[1] // 4
                    if self._pixel_shuffle
                    else image_embeddings.shape[1]
                )
                num_image_tiles = torch.full(
                    (image_embeddings.shape[0],),
                    tokens_per_tubelet,
                    dtype=torch.int,
                    device=image_embeddings.device,
                )""",
    ),
    (
        "use temporal post-resize dimensions for pixel shuffle",
        """                ps_h = ps_w = None
                if (
                    imgs_sizes is not None
                    and image_embeddings.shape[0] == 1
                    and imgs_sizes.shape[0] == 1
                ):
                    H = int(imgs_sizes[0, 0].item())
                    W = int(imgs_sizes[0, 1].item())
                    P = int(self.vision_model.patch_dim)
                    ps_h, ps_w = H // P, W // P""",
        """                ps_h = ps_w = None
                if use_temporal:
                    H = int(post_imgs_sizes[0, 0].item())
                    W = int(post_imgs_sizes[0, 1].item())
                    P = int(self.vision_model.patch_dim)
                    ps_h, ps_w = H // P, W // P
                elif (
                    imgs_sizes is not None
                    and image_embeddings.shape[0] == 1
                    and imgs_sizes.shape[0] == 1
                ):
                    H = int(imgs_sizes[0, 0].item())
                    W = int(imgs_sizes[0, 1].item())
                    P = int(self.vision_model.patch_dim)
                    ps_h, ps_w = H // P, W // P""",
    ),
)


def _info(msg: str) -> None:
    print(f"info: {msg}", file=sys.stderr)


def _cleanup_overlay_stubs(overlay_root: Path) -> None:
    for rel in _STALE_OVERLAY_INITS:
        path = overlay_root / rel
        if path.is_file():
            path.unlink()
            _info(f"removed stale overlay stub → {path}")
    for cache in overlay_root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)


def _recipe_fn_defined(recipe_source: Path, recipe_name: str) -> bool:
    return f"def {recipe_name}(" in (recipe_source / "recipe.py").read_text()


def _apply_llava_source_edits(source: str) -> str:
    for description, old, new in _LLAVA_SOURCE_REPLACEMENTS:
        matches = source.count(old)
        if matches != 1:
            raise ValueError(
                f"cannot apply LLaVA override: expected exactly one source block for "
                f"{description!r}, found {matches}"
            )
        source = source.replace(old, new, 1)
    return source


def _stage_llava_override(overlay_root: Path, megatron_lm_root: Path) -> None:
    if not (megatron_lm_root / ".git").exists():
        raise FileNotFoundError(f"Megatron-LM Git worktree not found: {megatron_lm_root}")

    pristine = subprocess.run(
        ["git", "show", f"HEAD:{LLAVA_MODEL_REL.as_posix()}"],
        cwd=megatron_lm_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    patched = _apply_llava_source_edits(pristine)

    dest = overlay_root / LLAVA_MODEL_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write(patched)
        os.replace(temp_name, dest)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    _info(f"staged LLaVA resize override → {dest}")


def deploy_overlay(
    overlay_root: Path,
    recipe_source: Path,
    integration_source: Path,
    recipe_name: str,
    megatron_lm_root: Path,
) -> None:
    if not _recipe_fn_defined(recipe_source, recipe_name):
        raise ValueError(
            f"recipe {recipe_name!r} is not defined in {recipe_source}/recipe.py; "
            "cannot stage AVLM training_data_processing modules for it",
        )

    overlay_root.mkdir(parents=True, exist_ok=True)
    _cleanup_overlay_stubs(overlay_root)
    _stage_llava_override(overlay_root, megatron_lm_root)

    sources = {"recipe": recipe_source}
    for key, rel_dest, names in _STAGE:
        dest = overlay_root / rel_dest
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = sources[key] / (f"{name}.py" if key == "recipe" else name)
            if not src.is_file():
                raise FileNotFoundError(f"missing module: {src}")
            shutil.copy2(src, dest / (f"{name}.py" if key == "recipe" else name))
            _info(f"staged {src.name} → {dest / src.name}")

    helper_dest = overlay_root / OVERLAY_RECIPE_DIR / "bridge_integration"
    helper_dest.mkdir(parents=True, exist_ok=True)
    (helper_dest / "__init__.py").touch(exist_ok=True)
    for name in (
        "task_encoder_collate.py",
        "hf_mcore_task_encoder.py",
        "megatron_fsdp_buffer_index.py",
        "megatron_fsdp_mamba_checkpoint.py",
        "peft_fsdp_pretrained_load.py",
        "nemotron_omni_audio_mask.py",
        "nemotron_omni_model_config.py",
        "nemotron_omni_squared_relu.py",
        "processed_prompt_log.py",
    ):
        shutil.copy2(integration_source / name, helper_dest / name)
        _info(f"staged {name} → {helper_dest / name}")

    training_dest = overlay_root / "megatron" / "bridge" / "training"
    if training_dest.is_dir():
        shutil.rmtree(training_dest)
        _info(f"removed stale overlay training tree → {training_dest}")

    _info(f"staged {recipe_name} overlay → {overlay_root}")


def main(argv: list[str] | None = None) -> int:
    integration = Path(__file__).resolve().parent
    recipe_source = integration.parent

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay-dir", type=Path, required=True)
    p.add_argument(
        "--recipe-name",
        required=True,
        help="Bridge recipe function name (from run.recipe in the training YAML)",
    )
    p.add_argument("--recipe-source", type=Path, default=recipe_source)
    p.add_argument("--integration-source", type=Path, default=integration)
    args = p.parse_args(argv)

    recipe_name = args.recipe_name.strip()
    recipe_src = args.recipe_source.resolve()

    if not _recipe_fn_defined(recipe_src, recipe_name):
        _info(
            f"skip overlay: {recipe_name!r} is not in {recipe_src}/recipe.py "
            "(using a built-in Bridge recipe)",
        )
        return 0

    bridge_root_text = os.environ.get("MEGATRON_BRIDGE_ROOT", "").strip()
    if not bridge_root_text:
        p.error("MEGATRON_BRIDGE_ROOT must be set")
    megatron_lm_root = Path(bridge_root_text).resolve() / "3rdparty" / "Megatron-LM"

    overlay_dir = args.overlay_dir.resolve()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    sentinel = overlay_dir.parent / f".overlay_ready.{os.environ.get('SLURM_JOB_ID', 'local')}"

    if rank != 0:
        while not sentinel.exists():
            time.sleep(2)
        return 0

    deploy_overlay(
        overlay_dir,
        recipe_src,
        args.integration_source.resolve(),
        recipe_name,
        megatron_lm_root,
    )
    if nnodes > 1:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
