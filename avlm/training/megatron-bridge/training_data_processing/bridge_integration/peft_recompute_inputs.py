"""Limit Bridge's PEFT recompute input-grad hook to trainable stacks."""

from __future__ import annotations

from functools import wraps
from typing import Iterable, Set

import torch


_PATCH_FLAG = "_avlm_targeted_peft_recompute_inputs_patch"


def _iter_unwrapped_models(model) -> Iterable[torch.nn.Module]:  # noqa: ANN001
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        yield from (module for module in unwrapped if module is not None)
    elif unwrapped is not None:
        yield unwrapped


def apply_peft_recompute_inputs_patch() -> None:
    """Only force input gradients on recomputed stacks with trainable parameters."""
    import megatron.bridge.peft.base as peft_base
    import megatron.bridge.peft.recompute as peft_recompute
    from megatron.bridge.utils.common_utils import print_rank_0
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.transformer_block import TransformerBlock

    if getattr(peft_recompute, _PATCH_FLAG, False):
        return

    stack_types = (TransformerBlock, HybridStack)

    def maybe_enable_recompute_inputs_grad(
        model,  # noqa: ANN001
        peft_recompute_patched: Set[int] | None = None,
    ) -> Set[int]:
        registry = (
            peft_recompute.PEFT_RECOMPUTE_PATCHED
            if peft_recompute_patched is None
            else peft_recompute_patched
        )
        patched_names = []

        try:
            for unwrapped_model in _iter_unwrapped_models(model):
                for module_name, module in unwrapped_model.named_modules():
                    if not isinstance(module, stack_types) or id(module) in registry:
                        continue

                    config = getattr(module, "config", None)
                    if (
                        config is None
                        or getattr(config, "recompute_granularity", None) != "full"
                        or getattr(config, "recompute_method", None) is None
                    ):
                        continue

                    if not any(parameter.requires_grad for parameter in module.parameters()):
                        continue

                    original_forward = module.forward

                    @wraps(original_forward)
                    def patched_forward(
                        hidden_states,
                        *args,
                        _original_forward=original_forward,
                        **kwargs,
                    ):
                        if (
                            torch.is_tensor(hidden_states)
                            and hidden_states.is_floating_point()
                            and not hidden_states.requires_grad
                        ):
                            hidden_states = hidden_states.detach().requires_grad_(True)
                        return _original_forward(hidden_states, *args, **kwargs)

                    module.forward = patched_forward
                    registry.add(id(module))
                    patched_names.append(f"{module_name or '<root>'} ({type(module).__name__})")
        except Exception as exc:  # pragma: no cover - best-effort compatibility patch
            print_rank_0(f"[AVLM PEFT+Recompute] Warning: failed to patch recomputed stacks: {exc}")

        if patched_names:
            print_rank_0(
                "[AVLM PEFT+Recompute] Enabled input gradients for trainable recomputed "
                f"stacks only: {', '.join(patched_names)}"
            )
        return registry

    peft_recompute.maybe_enable_recompute_inputs_grad = maybe_enable_recompute_inputs_grad
    peft_base.maybe_enable_recompute_inputs_grad = maybe_enable_recompute_inputs_grad
    setattr(peft_recompute, _PATCH_FLAG, True)


__all__ = ["apply_peft_recompute_inputs_patch"]
