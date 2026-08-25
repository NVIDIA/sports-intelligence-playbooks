"""Validation DP sharding with MoE-safe per-rank batch alignment."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Sampler


def batches_per_val_rank(num_examples: int, num_replicas: int) -> int:
    """Return the number of val batches every DP rank must run (MoE sync)."""
    if num_examples <= 0 or num_replicas <= 0:
        return 0
    return (num_examples + num_replicas - 1) // num_replicas


class ValShardSampler(Sampler[int]):
    """Assign each validation index to one DP rank, padding to a uniform batch count.

    Stride sharding keeps every example on exactly one rank (no cross-rank duplicates).
    When ``N`` is not divisible by ``world_size``, ranks pad by repeating their last
    assigned index so every rank executes the same number of forwards (required for
    MoE / DeepEP collectives). Idle ranks (``rank >= N``) repeat index ``0`` for sync.
    """

    def __init__(self, dataset: Sequence[Any], num_replicas: int, rank: int) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1 (got {num_replicas})")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}) (got {rank})")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        target = batches_per_val_rank(n, self.num_replicas)
        if target == 0:
            return iter(())
        if self.rank >= n:
            return iter([0] * target)
        indices = list(range(self.rank, n, self.num_replicas))
        if len(indices) < target:
            indices.extend([indices[-1]] * (target - len(indices)))
        return iter(indices)

    def __len__(self) -> int:
        return batches_per_val_rank(len(self.dataset), self.num_replicas)


def log_val_shard_layout(num_examples: int, num_replicas: int, rank: int) -> None:
    """Log how validation examples map to DP ranks (rank 0 only)."""
    import logging

    if rank != 0:
        return
    logger = logging.getLogger(__name__)
    if num_examples <= 0:
        logger.warning("Validation dataset is empty; val_loss will be undefined.")
        return
    batches = batches_per_val_rank(num_examples, num_replicas)
    active_ranks = min(num_examples, num_replicas)
    if num_examples < num_replicas:
        logger.warning(
            "Validation has %d example(s) and %d DP ranks: ranks 0–%d score one example "
            "each; ranks %d–%d run %d padded forward(s) on example 0 for MoE sync.",
            num_examples,
            num_replicas,
            active_ranks - 1,
            active_ranks,
            num_replicas - 1,
            batches,
        )
    elif num_examples % num_replicas != 0:
        logger.info(
            "Validation has %d example(s) across %d DP ranks (%d batches/rank; "
            "ranks with fewer examples pad by repeating their last shard for MoE sync).",
            num_examples,
            num_replicas,
            batches,
        )
