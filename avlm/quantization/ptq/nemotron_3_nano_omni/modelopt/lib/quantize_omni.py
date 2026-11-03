#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF PTQ + export for Nemotron-3-Nano-Omni (Model-Optimizer, LLM-only quant)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import logging
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint, save_expert_token_count_table
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from load_nano3omni import load_nano3omni
from quant_output import output_path, reset_output_path
from vlm_calib import get_jsonl_vlm_forward_loop

logger = logging.getLogger(__name__)

_HF_SIDECAR_FILENAMES = (
    "audio_model.py",
    "chat_template.jinja",
    "configuration_nemotron_h.py",
    "configuration.py",
    "configuration_radio.py",
    "evs.py",
    "image_processing.py",
    "modeling_nemotron_h.py",
    "modeling.py",
    "preprocessor_config.json",
    "processing.py",
    "processing_utils.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_io.py",
    "video_processing.py",
)

MIN_SAMPLES_NEED_FOR_CALIBRATION = 32
DEFAULT_NUM_SAMPLES = 512

_QUANT_CONFIG_BUILDERS: dict[str, Callable[[], dict]] = {}


def _has_local_checkpoint(root: Path) -> bool:
    if not root.is_dir():
        return False
    if (root / "model.safetensors.index.json").is_file():
        return True
    return any(root.glob("*.safetensors"))


def resolve_model_checkpoint_dir(model_path: str | Path) -> Path:
    """Local consolidated/ or Hub id → directory with weights (for file copies)."""
    raw = str(model_path)
    p = Path(raw)
    if _has_local_checkpoint(p):
        return p.resolve()
    if re.fullmatch(r"[^/]+/[^/]+", raw):
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=raw)
        return Path(local)
    return p


def _copy_hf_sidecars(src_root: Path, dest_root: Path) -> None:
    copied, missing = 0, 0
    for filename in _HF_SIDECAR_FILENAMES:
        src = src_root / filename
        if src.is_file():
            shutil.copy2(src, dest_root / filename)
            copied += 1
        else:
            missing += 1
            logger.warning("Sidecar missing on source checkpoint: %s", src)
    logger.info("Copied %d sidecar files from %s → %s (%d missing on source)", copied, src_root, dest_root, missing)


def _exclude_non_llm(quant_cfg: dict) -> None:
    for pattern in ["*vision*", "*image*", "*radio*", "*sound*", "*mlp1*"]:
        quant_cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})


def _get_fp8_default_llm_only() -> dict:
    quant_cfg = mtq.FP8_DEFAULT_CFG
    _exclude_non_llm(quant_cfg)
    return quant_cfg


def _get_nvfp4_default_nano3_llm_only() -> dict:
    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    _exclude_non_llm(quant_cfg)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})
    quant_cfg["quant_cfg"].append({"quantizer_name": "*mixer.conv1d*", "enable": False})
    for layer_idx in [4, 11, 18, 25, 32, 41]:
        quant_cfg["quant_cfg"].append({"quantizer_name": f"*layers.{layer_idx}.*", "enable": False})
    for layer_idx in [5, 12, 19, 26, 33, 42]:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            quant_cfg["quant_cfg"].append({"quantizer_name": f"*layers.{layer_idx}.mixer.{proj}*", "enable": False})
    return quant_cfg


def _get_nvfp4_experts_only() -> dict:
    from modelopt.torch.quantization.config import _nvfp4_selective_quant_cfg

    return _nvfp4_selective_quant_cfg(["*mixer.experts.*"])


def _set_kv_cache_constant_amax(quant_cfg: list) -> None:
    for i, entry in enumerate(quant_cfg):
        if entry.get("quantizer_name") != "*[kv]_bmm_quantizer":
            continue
        quant_cfg[i] = {**entry, "cfg": {**entry.get("cfg", {}), "use_constant_amax": True}}
        break


def _get_nvfp4_fp8_mixed() -> dict:
    from modelopt.torch.quantization.config import _nvfp4_cfg

    fp8_quantizer = {"num_bits": (4, 3), "axis": None}
    quant_cfg: list = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*mixer.experts.*weight_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*mixer.experts.*input_quantizer", "cfg": _nvfp4_cfg},
        {"quantizer_name": "*mixer.in_proj*weight_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.in_proj*input_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.out_proj*weight_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.out_proj*input_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.shared_experts*weight_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.shared_experts*input_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.o_proj*weight_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*mixer.o_proj*input_quantizer", "cfg": fp8_quantizer},
        {"quantizer_name": "*lm_head*", "enable": False},
        {"quantizer_name": "*output_layer*", "enable": False},
        {"quantizer_name": "*mixer.conv1d*", "enable": False},
        {"quantizer_name": "*mixer.gate*", "enable": False},
        {"quantizer_name": "*router*", "enable": False},
        {"quantizer_name": "*q_proj*", "enable": False},
        {"quantizer_name": "*k_proj*", "enable": False},
        {"quantizer_name": "*v_proj*", "enable": False},
        {"quantizer_name": "*fc1_latent_proj*", "enable": False},
        {"quantizer_name": "*fc2_latent_proj*", "enable": False},
        {"quantizer_name": "*vision*", "enable": False},
        {"quantizer_name": "*image*", "enable": False},
        {"quantizer_name": "*radio*", "enable": False},
        {"quantizer_name": "*sound*", "enable": False},
        {"quantizer_name": "*mlp1*", "enable": False},
    ]
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def _get_fp8_default_llm_only_fp8_kv() -> dict:
    import copy

    quant_cfg = copy.deepcopy(_get_fp8_default_llm_only())
    quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(quant_cfg, mtq.FP8_KV_CFG["quant_cfg"])
    _set_kv_cache_constant_amax(quant_cfg["quant_cfg"])
    return quant_cfg


def _get_nvfp4_fp8_mixed_fp8_kv() -> dict:
    import copy

    quant_cfg = copy.deepcopy(_get_nvfp4_fp8_mixed())
    quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(quant_cfg, mtq.FP8_KV_CFG["quant_cfg"])
    _set_kv_cache_constant_amax(quant_cfg["quant_cfg"])
    return quant_cfg


def _register_quant_builders() -> None:
    global _QUANT_CONFIG_BUILDERS
    _QUANT_CONFIG_BUILDERS = {
        "fp8_default_llm_only": _get_fp8_default_llm_only,
        "fp8": _get_fp8_default_llm_only,
        "fp8_default_llm_only_fp8_kv": _get_fp8_default_llm_only_fp8_kv,
        "fp8_fp8_kv": _get_fp8_default_llm_only_fp8_kv,
        "nvfp4_default_nano3_llm_only": _get_nvfp4_default_nano3_llm_only,
        "nvfp4": _get_nvfp4_default_nano3_llm_only,
        "nvfp4_experts_only": _get_nvfp4_experts_only,
        "nvfp4_fp8_mixed": _get_nvfp4_fp8_mixed,
        "nvfp4_fp8_mixed_fp8_kv": _get_nvfp4_fp8_mixed_fp8_kv,
    }


# Processor metadata; not accepted by NemotronHForCausalLM.generate (see repo inference scripts).
_PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")


def get_forward_loop(model, processor, tokenizer, num_samples=32):
    calib_dataloader = get_vlm_dataset_dataloader(
        dataset_name="nemotron_vlm_dataset_v2",
        processor=processor,
        batch_size=1,
        num_samples=num_samples,
        device=model.device,
        require_image=True,
        subsets=["sparsetables", "plotqa_cot", "wiki_en"],
        shuffle_buffer_size=10_000,
        seed=42,
        use_media_shards=True,
        max_shards=1,
    )

    calib_stats = {"completed": 0}

    def forward_loop():
        for i, batch in enumerate(calib_dataloader):
            calib_stats["completed"] = i + 1
            logger.info(
                "Running calibration forward %d (target %d VLM samples)",
                calib_stats["completed"],
                num_samples,
            )
            _flush_log_handlers()
            for key in _PROCESSOR_METADATA_KEYS:
                batch.pop(key, None)
            with torch.no_grad():
                model.generate(
                    **batch,
                    max_new_tokens=4,
                    do_sample=False,
                    output_hidden_states=False,
                    eos_token_id=tokenizer.eos_token_id,
                )
        if calib_stats["completed"] < num_samples:
            logger.warning(
                "Calibration dataloader yielded %d/%d image-text samples (subset/shard limits)",
                calib_stats["completed"],
                num_samples,
            )

    forward_loop.calib_stats = calib_stats
    return forward_loop


def _get_quant_config(name: str) -> dict:
    builder = _QUANT_CONFIG_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown quantize_config: {name!r}. Choose from {sorted(_QUANT_CONFIG_BUILDERS)}")
    return builder()


def strip_parakeet_conv_bias(src_path: Path) -> int:
    src_path = Path(src_path)
    index_path = src_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        single_shard = src_path / "model.safetensors"
        with safe_open(str(single_shard), framework="pt") as f:
            keys = list(f.keys())
        index = {"metadata": {"total_size": 0}, "weight_map": dict.fromkeys(keys, "model.safetensors")}
    weight_map = index["weight_map"]

    pattern = re.compile(
        r"^sound_encoder\.encoder\.layers\.\d+\.conv\.(depthwise_conv|pointwise_conv1|pointwise_conv2)\.bias$"
    )
    keys_to_strip = sorted(k for k in weight_map if pattern.match(k))
    if not keys_to_strip:
        logger.info("%s: no parakeet conv bias keys to strip", src_path)
        return 0

    shards_to_rewrite: dict[str, list[str]] = {}
    for key in keys_to_strip:
        shards_to_rewrite.setdefault(weight_map[key], []).append(key)

    total_stripped_bytes = 0
    for shard_name, keys_in_shard in shards_to_rewrite.items():
        src_shard = src_path / shard_name
        logger.info("Rewriting %s, stripping %d keys", shard_name, len(keys_in_shard))
        tensors = {}
        with safe_open(str(src_shard), framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                if k in keys_in_shard:
                    t = f.get_tensor(k)
                    total_stripped_bytes += t.numel() * t.element_size()
                    continue
                tensors[k] = f.get_tensor(k)
        save_file(tensors, str(src_shard), metadata=metadata)

    for key in keys_to_strip:
        del weight_map[key]
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] -= total_stripped_bytes
    with open(src_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    logger.info("Stripped %d parakeet conv bias keys (%d bytes)", len(keys_to_strip), total_stripped_bytes)
    return len(keys_to_strip)


def _rename_weight_map_key(key: str) -> str:
    """Map legacy ModelOpt export names to Nemotron-Omni HF / vLLM layout."""
    if key.startswith("language_backbone.backbone"):
        return "language_model.backbone" + key[len("language_backbone.backbone") :]
    if key.startswith("language_backbone."):
        return "language_model." + key[len("language_backbone.") :]
    return key


def normalize_language_module_prefixes(src_path: Path) -> int:
    """Rewrite language_backbone.* tensor keys → language_model.* (shard files + sidecars)."""
    src_path = Path(src_path)
    index_path = src_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    else:
        single_shard = src_path / "model.safetensors"
        if not single_shard.is_file():
            return 0
        with safe_open(str(single_shard), framework="pt") as f:
            keys = list(f.keys())
        weight_map = dict.fromkeys(keys, "model.safetensors")
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}

    rename_pairs = [(k, _rename_weight_map_key(k)) for k in weight_map if k.startswith("language_backbone.")]
    if not rename_pairs:
        return 0

    shards_to_rewrite: dict[str, list[tuple[str, str]]] = {}
    new_weight_map = dict(weight_map)
    for old_key, new_key in rename_pairs:
        shard = weight_map[old_key]
        shards_to_rewrite.setdefault(shard, []).append((old_key, new_key))
        del new_weight_map[old_key]
        new_weight_map[new_key] = shard

    for shard_name, pairs in shards_to_rewrite.items():
        shard_path = src_path / shard_name
        logger.info("Renaming %d language_backbone keys in %s", len(pairs), shard_name)
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt") as f:
            metadata = f.metadata() or {}
            rename_old = {old for old, _ in pairs}
            pair_map = dict(pairs)
            for k in f.keys():
                tensors[pair_map[k] if k in rename_old else k] = f.get_tensor(k)
        save_file(tensors, str(shard_path), metadata=metadata)

    index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    quant_cfg = src_path / "hf_quant_config.json"
    if quant_cfg.is_file():
        text = quant_cfg.read_text()
        text = text.replace("language_backbone.backbone", "language_model.backbone")
        text = text.replace("language_backbone.", "language_model.")
        quant_cfg.write_text(text)

    for json_name in ("config.json", "config.json.bak"):
        cfg_path = src_path / json_name
        if cfg_path.is_file():
            text = cfg_path.read_text()
            text = text.replace("language_backbone.backbone", "language_model.backbone")
            text = text.replace("language_backbone.", "language_model.")
            cfg_path.write_text(text)

    logger.info(
        "Normalized %d export keys: language_backbone.* → language_model.* (%s)",
        len(rename_pairs),
        src_path,
    )
    return len(rename_pairs)


def _token_ids_from_generate(generate_out) -> torch.Tensor:
    """Normalize HF generate() return value for batch_decode (tensor vs GenerateOutput vs dict)."""
    if isinstance(generate_out, torch.Tensor):
        return generate_out
    if hasattr(generate_out, "sequences"):
        return generate_out.sequences
    if isinstance(generate_out, dict):
        if "sequences" in generate_out:
            return generate_out["sequences"]
        if "input_ids" in generate_out:
            return generate_out["input_ids"]
    raise TypeError(f"Unexpected generate() return type: {type(generate_out)!r}")


def _run_sample_text_inference(model, tokenizer) -> None:
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    try:
        generate_out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )
        output_ids = _token_ids_from_generate(generate_out)
        logger.info("Sample output: %s", tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0])
    except Exception as exc:
        logger.warning("Post-quant smoke generate skipped (%s: %s)", type(exc).__name__, exc)


def _log_package_versions() -> None:
    for pkg in ("nvidia-modelopt", "transformers", "torch"):
        try:
            logger.info("%s version: %s", pkg, version(pkg))
        except PackageNotFoundError:
            logger.info("%s: not installed", pkg)


def _flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _setup_job_logging(log_dir: Path) -> None:
    """Configure root + file logging after heavy imports (basicConfig alone may be a no-op)."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    basic_kwargs: dict = {"level": logging.INFO, "format": log_format}
    try:
        logging.basicConfig(**basic_kwargs, force=True)
    except TypeError:
        logging.basicConfig(**basic_kwargs)
    root.setLevel(logging.INFO)

    log_file = log_dir / "run.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    root.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info("Logging to %s", log_file)
    _flush_log_handlers()


def main() -> None:
    _register_quant_builders()
    parser = argparse.ArgumentParser(description="Nemotron-3-Nano-Omni HF PTQ")
    parser.add_argument(
        "--quantize-config",
        type=str,
        default="fp8_fp8_kv",
        choices=sorted(_QUANT_CONFIG_BUILDERS),
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--model-path", type=str, required=True, help="HF Hub id or local checkpoint directory")
    parser.add_argument(
        "--calib-jsonl",
        type=str,
        default="",
        help="Sports video JSONL for calibration; omit for nemotron_vlm_dataset_v2",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="",
        help="Root directory for relative video paths in --calib-jsonl rows",
    )
    parser.add_argument("--calib-video-fps", type=int, default=2)
    parser.add_argument("--calib-max-video-frames", type=int, default=128)
    parser.add_argument(
        "--video-pruning-rate",
        type=float,
        default=0.0,
        help="EVS video token pruning in generate (0 = off; matches inference when 0)",
    )
    args = parser.parse_args()

    reset_output_path()
    log_dir = output_path()
    _setup_job_logging(log_dir)

    _log_package_versions()
    logger.info("Args: %s", vars(args))

    ckpt_dir = resolve_model_checkpoint_dir(args.model_path)
    logger.info("Model checkpoint dir for sidecars/weights: %s", ckpt_dir)
    model, processor, tokenizer = load_nano3omni(
        args.model_path,
        video_pruning_rate=args.video_pruning_rate,
    )

    if args.num_samples < MIN_SAMPLES_NEED_FOR_CALIBRATION:
        logger.warning(
            "num_samples %d < %d — MoE experts may not all calibrate",
            args.num_samples,
            MIN_SAMPLES_NEED_FOR_CALIBRATION,
        )

    quant_cfg = _get_quant_config(args.quantize_config)
    if args.calib_jsonl:
        if not args.video_root:
            raise SystemExit("--video-root is required when --calib-jsonl is set")
        forward_loop = get_jsonl_vlm_forward_loop(
            model,
            processor,
            tokenizer,
            calib_jsonl=Path(args.calib_jsonl),
            video_root=Path(args.video_root),
            num_samples=args.num_samples,
            video_fps=args.calib_video_fps,
            max_video_frames=args.calib_max_video_frames,
        )
    else:
        forward_loop = get_forward_loop(model, processor, tokenizer, num_samples=args.num_samples)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    completed = forward_loop.calib_stats["completed"]
    logger.info(
        "PTQ calibration complete (%d/%d VLM forwards). Next: quant summary, smoke generate, HF export.",
        completed,
        args.num_samples,
    )
    _flush_log_handlers()

    summary_root = str(log_dir).rstrip("/")
    mtq.print_quant_summary(model, summary_root)
    save_expert_token_count_table(model, summary_root)

    checkpoint_path = log_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    extra_state_dict = {}
    vid_key = "vision_model.radio_model.model.patch_generator.video_embedder.weight"
    src_index_path = ckpt_dir / "model.safetensors.index.json"
    src_single_path = ckpt_dir / "model.safetensors"
    if src_index_path.exists():
        with open(src_index_path) as f:
            src_index = json.load(f)
        src_shard = src_index.get("weight_map", {}).get(vid_key)
        if src_shard:
            extra_state_dict[vid_key] = load_file(str(ckpt_dir / src_shard))[vid_key]
    elif src_single_path.exists():
        src_tensors = load_file(str(src_single_path))
        if vid_key in src_tensors:
            extra_state_dict[vid_key] = src_tensors[vid_key]

    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=str(checkpoint_path),
        extra_state_dict=extra_state_dict or None,
    )

    _copy_hf_sidecars(ckpt_dir, checkpoint_path)

    normalize_language_module_prefixes(checkpoint_path)

    config_path = checkpoint_path / "config.json"
    shutil.copy2(config_path, checkpoint_path / "config.json.bak")
    config_text = config_path.read_text()
    config_text = config_text.replace("language_model.backbone", "language_model.model")
    config_text = config_text.replace('"language_model.model.layers.4"', '"language_model.model.layers.4."')
    config_text = config_text.replace('"language_model.model.layers.5"', '"language_model.model.layers.5."')
    config_path.write_text(config_text)
    with open(config_path) as f:
        export_cfg = json.load(f)
    if export_cfg.get("video_pruning_rate") != args.video_pruning_rate:
        export_cfg["video_pruning_rate"] = args.video_pruning_rate
        with open(config_path, "w") as f:
            json.dump(export_cfg, f, indent=2)
            f.write("\n")
        logger.info("Exported config.json: video_pruning_rate=%s", args.video_pruning_rate)

    strip_parakeet_conv_bias(checkpoint_path)
    logger.info(
        "HF export finished (weights + configs under %s). quantize.sh will publish to HF_QUANT_EXPORT_PATH.",
        checkpoint_path,
    )
    _run_sample_text_inference(model, tokenizer)
    _flush_log_handlers()


if __name__ == "__main__":
    main()
