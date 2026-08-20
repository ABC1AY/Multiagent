"""Shared type definitions for GRPO training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LeaderAction:
    """A single action emitted by the scheduling Leader."""
    action_type: str
    worker_id: int | None = None
    chunk_id: int | None = None
    query_text: str | None = None
    arbitration_chunk: str | None = None
    raw_text: str | None = None
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None


@dataclass
class Episode:
    """One rollout trajectory used for GRPO training."""
    question: str
    reference_answer: str
    final_prediction: str
    total_token_cost: int
    total_worker_calls: int
    sequence_log_prob: float
    reward: float = 0.0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
