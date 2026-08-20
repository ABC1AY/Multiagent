"""Reward functions for GRPO training."""
from __future__ import annotations

import re
import string


def _normalize_answer(text: str) -> str:
    """Normalize answer text for matching."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", "", text)
    return text


def compute_reward(
    prediction: str,
    reference: str,
    token_cost: int,
    worker_calls: int,
    accuracy_weight: float = 1.0,
    efficiency_weight: float = 0.1,
) -> float:
    """Compute a scalar reward combining answer accuracy and efficiency cost."""
    pred = _normalize_answer(prediction)
    ref = _normalize_answer(reference)
    # Consider correct if either string contains the other (handles short answers).
    accuracy = 1.0 if (ref in pred or pred in ref) else 0.0
    cost_penalty = efficiency_weight * (token_cost / 1000.0 + worker_calls / 10.0)
    return accuracy_weight * accuracy - cost_penalty


def compute_group_relative_advantages(rewards: list[float]) -> list[float]:
    """Convert raw rewards into group-relative advantages."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    if std == 0:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]
