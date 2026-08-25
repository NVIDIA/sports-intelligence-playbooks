# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Read YAML and print shell export KEY=value lines for sourcing.
# Lists become space-separated values (for SLURM_ACCOUNTS / SLURM_ACCOUNTS_RACE). Booleans become 0/1.
#
# Usage:
#   # emit all top-level keys
#   _load_params_yaml.py path/to/params.yaml
#
#   # emit selected nested keys via VAR=dotted.path specs
#   _load_params_yaml.py path/to/recipe.yaml \
#       TP=distributed.tp_size PP=distributed.pp_size EP=distributed.ep_size

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HYPERPARAM_SKIP_KEYS = frozenset({"trials", "hyperparam_name"})


def _format_slurm_time(value: Any) -> str:
    """Slurm --time accepts H:MM:SS; YAML parses unquoted 4:00:00 as sexagesimal int (14400)."""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and value >= 3600 and value % 60 == 0:
        hours, rem = divmod(value, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return str(value)


def _emit(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        text = "1" if value else "0"
    elif isinstance(value, (list, tuple)):
        text = " ".join(str(item) for item in value)
    elif key == "SLURM_TIME_LIMIT":
        text = _format_slurm_time(value)
    else:
        text = str(value)
    # shell-safe: wrap in double quotes; escape embedded quotes and backslashes
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    print(f'export {key}="{escaped}"')


def _resolve_dotted(root: Any, dotted: str) -> Any:
    cur = root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _emit_hyperparam_defaults(cfg: dict[str, Any]) -> int:
    for key, value in cfg.items():
        if not isinstance(key, str) or key.startswith("_") or key in _HYPERPARAM_SKIP_KEYS:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        _emit(key, value)
    return 0


def _emit_hyperparam_trials(cfg: dict[str, Any]) -> int:
    trials = cfg.get("trials")
    if not isinstance(trials, list) or not trials:
        print(
            "error: hyperparameter-search YAML must define a non-empty trials list",
            file=sys.stderr,
        )
        return 1
    for i, trial in enumerate(trials):
        if not isinstance(trial, dict):
            print(f"error: trials[{i}] must be a mapping", file=sys.stderr)
            return 1
        name = trial.get("name")
        if not isinstance(name, str) or not name.strip():
            print(f"error: trials[{i}] missing name", file=sys.stderr)
            return 1
        config_path = trial.get("config")
        if not isinstance(config_path, str) or not config_path.strip():
            print(f"error: trials[{i}] missing config", file=sys.stderr)
            return 1

        print(f"{name.strip()}\t{config_path.strip()}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: _load_params_yaml.py <yaml> [VAR=dotted.path ...]",
            file=sys.stderr,
        )
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1
    try:
        import yaml
    except ImportError:
        print("error: PyYAML required to load YAML", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        return 0
    if not isinstance(cfg, dict):
        print("error: YAML root must be a mapping", file=sys.stderr)
        return 1

    specs = sys.argv[2:]
    if specs == ["--hyperparam-defaults"]:
        return _emit_hyperparam_defaults(cfg)
    if specs == ["--hyperparam-trials"]:
        return _emit_hyperparam_trials(cfg)

    if specs:
        for spec in specs:
            if "=" not in spec:
                print(f"error: bad selector (expected VAR=dotted.path): {spec}", file=sys.stderr)
                return 2
            var, dotted = spec.split("=", 1)
            if not var or not dotted:
                print(f"error: bad selector (expected VAR=dotted.path): {spec}", file=sys.stderr)
                return 2
            _emit(var, _resolve_dotted(cfg, dotted))
        return 0

    for key, value in cfg.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        _emit(key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
