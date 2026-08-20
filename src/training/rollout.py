"""Rollout collection and recording for GRPO."""
from __future__ import annotations

import logging
from typing import Any

from src.agents.worker import Worker
from src.training.types import Episode, LeaderAction
from src.training.reward import compute_reward

logger = logging.getLogger(__name__)


class RolloutRecorder:
    """Records the scheduling trajectory of a single question."""

    def __init__(self, question: str, reference_answer: str, chunks: list[str]):
        self.question = question
        self.reference_answer = reference_answer
        self.chunks = chunks
        self.trajectory: list[dict[str, Any]] = []

    def record_action(
        self,
        action_type: str,
        worker_id: int | None = None,
        chunk_id: int | None = None,
        query_text: str | None = None,
        token_ids: list[int] | None = None,
        log_probs: list[float] | None = None,
        raw_text: str | None = None,
    ) -> None:
        """Record a generic scheduling action."""
        action = LeaderAction(
            action_type=action_type,
            worker_id=worker_id,
            chunk_id=chunk_id,
            query_text=query_text,
            token_ids=token_ids,
            log_probs=log_probs,
            raw_text=raw_text,
            arbitration_chunk=(
                self.chunks[chunk_id] if chunk_id is not None and chunk_id < len(self.chunks) else None
            ),
        )
        self.trajectory.append({"type": "action", "action": action})

    def record_conflict_resolution(
        self,
        state: dict[str, Any],
        swap_wid_a: int,
        swap_wid_b: int,
        swap_resp_a: str,
        swap_resp_b: str,
        swap_resp_a_token_ids: list[int],
        swap_resp_a_log_probs: list[float],
        swap_resp_b_token_ids: list[int],
        swap_resp_b_log_probs: list[float],
        arb_wid: int | None,
        arb_resp: str | None,
        arb_resp_token_ids: list[int] | None,
        arb_resp_log_probs: list[float] | None,
    ) -> None:
        """Record a conflict-resolution step for GRPO training."""
        swap_action_a = LeaderAction(
            action_type="SWAP",
            worker_id=swap_wid_a,
            chunk_id=swap_wid_b,
            query_text=swap_resp_a,
            arbitration_chunk=(
                self.chunks[swap_wid_b] if swap_wid_b < len(self.chunks) else None
            ),
            token_ids=swap_resp_a_token_ids,
            log_probs=swap_resp_a_log_probs,
        )
        swap_action_b = LeaderAction(
            action_type="SWAP",
            worker_id=swap_wid_b,
            chunk_id=swap_wid_a,
            query_text=swap_resp_b,
            arbitration_chunk=(
                self.chunks[swap_wid_a] if swap_wid_a < len(self.chunks) else None
            ),
            token_ids=swap_resp_b_token_ids,
            log_probs=swap_resp_b_log_probs,
        )
        arb_action: LeaderAction | None = None
        if arb_wid is not None and arb_resp is not None:
            arb_action = LeaderAction(
                action_type="ARBITRATE",
                worker_id=arb_wid,
                query_text=arb_resp,
                arbitration_chunk=(
                    f"{self.chunks[swap_wid_a]}\n\n{self.chunks[swap_wid_b]}"
                ),
                token_ids=arb_resp_token_ids,
                log_probs=arb_resp_log_probs,
            )
        self.trajectory.append({
            "type": "conflict_resolution",
            "state": state,
            "swap_actions": [swap_action_a, swap_action_b],
            "arb_action": arb_action,
        })


def _conflict_log_prob_sum(trajectory_entry: dict[str, Any]) -> float:
    """Sum log-probs of the swap + arbitration actions in a conflict-resolution step."""
    total = 0.0
    for action in trajectory_entry.get("swap_actions", []):
        if action.log_probs:
            total += sum(lp.item() for lp in action.log_probs)
    arb_action = trajectory_entry.get("arb_action")
    if arb_action is not None and arb_action.log_probs:
        total += sum(lp.item() for lp in arb_action.log_probs)
    return total


def generate_rollout_batch(
    leader: "Leader",
    question: str,
    reference_answer: str,
    chunks: list[str],
    workers: list[Worker],
    group_size: int = 4,
    top_k: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[Episode]:
    """Collect a group of rollout trajectories for GRPO.

    Each trajectory is a sampled run of the agent scheduling loop.  Rewards are
    computed immediately so that the trainer can form a group baseline.
    """
    episodes: list[Episode] = []
    for _ in range(group_size):
        recorder = RolloutRecorder(question, reference_answer, chunks)
        result = leader.run_agent_loop_with_logprobs(
            question=question,
            chunks=chunks,
            workers=workers,
            recorder=recorder,
            seed_top_k=top_k,
            temperature=temperature,
            top_p=top_p,
        )
        episode = Episode(
            question=question,
            reference_answer=reference_answer,
            final_prediction=result["final_answer"],
            total_token_cost=result["total_token_cost"],
            total_worker_calls=result["worker_calls"],
            sequence_log_prob=result["sequence_log_prob"],
            trajectory=recorder.trajectory,
        )
        episode.reward = compute_reward(
            prediction=episode.final_prediction,
            reference=episode.reference_answer,
            token_cost=episode.total_token_cost,
            worker_calls=episode.total_worker_calls,
        )
        episodes.append(episode)
    return episodes
