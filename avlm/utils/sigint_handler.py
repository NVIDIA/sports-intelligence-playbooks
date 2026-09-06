# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive Ctrl+C handling for torchrun worker processes."""

from __future__ import annotations

import logging
import os
import signal
from typing import Any

logger = logging.getLogger(__name__)

_REGISTERED = False


def _ctrl_c_stops_immediately() -> bool:
    """True when Ctrl+C should exit the worker promptly (interactive/local runs)."""
    kind = os.environ.get("TRAIN_LAUNCH_KIND", "")
    if kind in ("interactive", "local"):
        return True
    if kind == "batch":
        return False
    return os.environ.get("INSIDE_INTERACTIVE_SESSION", "0") == "1"


def register_interactive_sigint_handler() -> None:
    """Handle Ctrl+C (SIGINT) inside torchrun workers.

    ``StepScheduler`` only wires SIGTERM (Slurm ``scancel``). Interactive runs need
    SIGINT as well. Shell ``train_run_with_interrupt`` kills the full torchrun tree
    when ranks are stuck in NCCL; this handler covers the common case where the
    main thread is in Python.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from nemo_automodel.components.training import step_scheduler as step_scheduler_mod

    if getattr(step_scheduler_mod.StepScheduler.__init__, "_avlm_sigint_patched", False):
        _REGISTERED = True
        return

    _orig_init = step_scheduler_mod.StepScheduler.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        sig_handler = self.sig_handler

        def _on_sigint(signum: int, frame: Any) -> None:
            sig_handler._signal_received = True
            if not _ctrl_c_stops_immediately():
                logger.info("Received SIGINT (Ctrl+C), graceful stop after this step")
                return
            logger.info("Received SIGINT (Ctrl+C), stopping now")
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, _on_sigint)

    _patched_init._avlm_sigint_patched = True  # type: ignore[attr-defined]
    step_scheduler_mod.StepScheduler.__init__ = _patched_init
    _REGISTERED = True
