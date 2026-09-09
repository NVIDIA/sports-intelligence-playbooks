# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re

import yaml


def suite_runs(cfg):
    items = cfg.get("runs")
    if not isinstance(items, list) or not items:
        raise ValueError("suite YAML must define a non-empty runs list")
    for i, run in enumerate(items):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{i}] must be a mapping")
        name = run.get("INFERENCE_NAME")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"runs[{i}] missing INFERENCE_NAME")
    return items


def value_text(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        raise ValueError("suite values must be scalar")
    return str(value)


def print_args(cfg, run):
    args = {}
    for key, value in cfg.items():
        if key not in {"suite_name", "runs"}:
            args[key] = value_text(value)
    for key, value in run.items():
        args[key] = value_text(value)
    for key, value in args.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"suite key is not a valid environment variable: {key}")
        if value is not None:
            print(f"{key}={value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config")
    parser.add_argument("command", choices=("name", "count", "run-name", "args"))
    parser.add_argument("run_idx", nargs="?", type=int)
    args = parser.parse_args()

    try:
        with open(args.suite_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("suite YAML root must be a mapping")

        if args.command == "name":
            print(str(cfg.get("suite_name") or "").strip())
        else:
            runs = suite_runs(cfg)
            if args.command == "count":
                print(len(runs))
            elif args.run_idx is None:
                parser.error(f"{args.command} requires run_idx")
            elif args.command == "run-name":
                print(str(runs[args.run_idx]["INFERENCE_NAME"]).strip())
            elif args.command == "args":
                print_args(cfg, runs[args.run_idx])
            else:
                raise RuntimeError(f"unknown command: {args.command}")
    except Exception as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
