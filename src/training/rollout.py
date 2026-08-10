"""GRPO 训练的 Episode 轨迹记录器。

提供 ``RolloutRecorder`` 类和 ``run_rollout_with_trajectory()`` 函数，
用于在 multi-agent pipeline 执行过程中记录完整的 (state, action, log-prob, reward) 轨迹。
"""
import uuid

from src.agents.leader import Leader
from src.agents.worker import NO_MENTION_MARKER, Worker
from src.models.model_loader import generate_with_logprobs
from src.retrieval.bm25_retriever import BM25Retriever
from src.training.reward import compute_reward
from src.training.types import (
    DecisionPoint,
    Episode,
    LeaderAction,
    WorkerAction,
)


class RolloutRecorder:
    """在多智能体执行过程中记录 episode 轨迹。"""

    def __init__(self):
        self.current_episode: Episode | None = None
        self.step_counter = 0

    def start_episode(
        self,
        question: str,
        reference_answer: str,
        document_chunks: list[str],
    ) -> Episode:
        """初始化一个新的 episode。"""
        self.current_episode = Episode(
            episode_id=str(uuid.uuid4()),
            question=question,
            reference_answer=reference_answer,
            document_chunks=document_chunks,
        )
        self.step_counter = 0
        return self.current_episode

    def record_worker_query(
        self,
        worker_id: int,
        chunk_index: int,
        question: str,
        chunk_text: str,
        response_text: str,
        response_token_ids: list[int],
        response_log_probs: list[float],
    ) -> None:
        """记录一次 Worker 查询动作。"""
        if self.current_episode is None:
            raise RuntimeError("没有活动的 episode")

        action = WorkerAction(
            worker_id=worker_id,
            chunk_index=chunk_index,
            question=question,
            chunk_text=chunk_text,
            response_text=response_text,
            response_token_ids=response_token_ids,
            response_log_probs=response_log_probs,
        )

        state = {
            "question": question,
            "chunk_index": chunk_index,
            "accumulated_responses": [
                (d.action.worker_id, d.action.response_text)
                for d in self.current_episode.decisions
                if d.action_type == "worker_query"
            ],
        }

        decision = DecisionPoint(
            step=self.step_counter,
            state=state,
            action=action,
            action_type="worker_query",
        )

        self.current_episode.decisions.append(decision)
        self.step_counter += 1

    def record_leader_synthesis(
        self,
        worker_responses_used: list[int],
        generated_text: str,
        generated_token_ids: list[int],
        generated_log_probs: list[float],
    ) -> None:
        """记录 Leader 的综合动作。"""
        if self.current_episode is None:
            raise RuntimeError("没有活动的 episode")

        action = LeaderAction(
            action_type="synthesis",
            worker_responses_used=worker_responses_used,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            generated_log_probs=generated_log_probs,
        )

        state = {
            "question": self.current_episode.question,
            "all_worker_responses": [
                (d.action.worker_id, d.action.response_text)
                for d in self.current_episode.decisions
                if d.action_type == "worker_query"
            ],
        }

        decision = DecisionPoint(
            step=self.step_counter,
            state=state,
            action=action,
            action_type="leader_synthesis",
        )

        self.current_episode.decisions.append(decision)
        self.step_counter += 1

    def finalize_episode(
        self,
        final_prediction: str,
        total_token_cost: int,
        total_worker_calls: int,
    ) -> Episode:
        """完成 episode 并计算奖励。"""
        if self.current_episode is None:
            raise RuntimeError("没有活动的 episode")

        self.current_episode.final_prediction = final_prediction
        self.current_episode.total_token_cost = total_token_cost
        self.current_episode.total_worker_calls = total_worker_calls

        # 计算奖励
        self.current_episode.reward = compute_reward(
            prediction=final_prediction,
            reference=self.current_episode.reference_answer,
            token_cost=total_token_cost,
            worker_calls=total_worker_calls,
        )

        episode = self.current_episode
        self.current_episode = None
        return episode


def run_rollout_with_trajectory(
    leader: Leader,
    question: str,
    reference_answer: str,
    chunks: list[str],
    workers: list[Worker],
    top_k: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_worker_tokens: int = 64,
    max_leader_tokens: int = 128,
) -> Episode:
    """执行单次 rollout 并记录完整轨迹。

    这是 ``Leader.run_selective_loop()`` 的训练兼容版本。
    使用采样（``do_sample=True``）并记录所有决策及其对数概率。

    Args:
        leader: Leader 智能体。
        question: 要回答的问题。
        reference_answer: 用于计算奖励的参考答案。
        chunks: 文档片段列表。
        workers: Worker 智能体列表（每个 chunk 一个）。
        top_k: BM25 检索的 chunk 数量。
        temperature: 采样温度。
        top_p: nucleus 采样的概率阈值。
        max_worker_tokens: Worker 最大生成 token 数。
        max_leader_tokens: Leader 最大生成 token 数。

    Returns:
        包含完整轨迹的 Episode 对象。
    """
    assert len(chunks) == len(workers), "chunks 和 workers 数量必须一致"
    assert top_k <= len(chunks), "top_k 不能超过 chunk 总数"

    recorder = RolloutRecorder()
    recorder.start_episode(question, reference_answer, chunks)

    # BM25 检索
    retriever = BM25Retriever(leader.tokenizer).fit(chunks)
    top_results = retriever.retrieve(question, top_k=top_k)
    top_indices = [idx for idx, _ in top_results]

    # 使用采样和对数概率查询选中的 Worker
    worker_responses: list[tuple[int, str]] = []
    total_token_cost = 0
    total_worker_calls = 0

    for idx in top_indices:
        worker = workers[idx]
        result = worker.answer_with_logprobs(
            question,
            chunks[idx],
            max_new_tokens=max_worker_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        recorder.record_worker_query(
            worker_id=worker.worker_id,
            chunk_index=idx,
            question=question,
            chunk_text=chunks[idx],
            response_text=result["text"],
            response_token_ids=result["token_ids"],
            response_log_probs=result["log_probs"],
        )

        worker_responses.append((worker.worker_id, result["text"]))
        total_worker_calls += 1
        # 近似 token 消耗：prompt (~100) + response tokens
        total_token_cost += len(result["token_ids"]) + 100

    # Leader 综合（使用采样和对数概率）
    synthesis_messages = leader._build_synthesis_prompt(question, worker_responses)
    final_text, final_token_ids, final_log_probs = generate_with_logprobs(
        leader.model,
        leader.tokenizer,
        synthesis_messages,
        max_new_tokens=max_leader_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    recorder.record_leader_synthesis(
        worker_responses_used=[wid for wid, _ in worker_responses],
        generated_text=final_text.strip(),
        generated_token_ids=final_token_ids,
        generated_log_probs=final_log_probs,
    )

    # 近似 token 消耗：synthesis prompt (~200) + response tokens
    total_token_cost += len(final_token_ids) + 200

    episode = recorder.finalize_episode(
        final_prediction=final_text.strip(),
        total_token_cost=total_token_cost,
        total_worker_calls=total_worker_calls,
    )

    return episode


def generate_rollout_batch(
    leader: Leader,
    question: str,
    reference_answer: str,
    chunks: list[str],
    workers: list[Worker],
    group_size: int = 4,
    top_k: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[Episode]:
    """为同一 prompt 生成 G 个独立 rollout（GRPO 的 group）。

    Args:
        leader: Leader 智能体。
        question: 要回答的问题。
        reference_answer: 参考答案。
        chunks: 文档片段列表。
        workers: Worker 智能体列表。
        group_size: 每组 rollout 的数量 (G)。
        top_k: BM25 检索的 chunk 数量。
        temperature: 采样温度。
        top_p: nucleus 采样阈值。

    Returns:
        G 个 Episode 对象的列表。
    """
    episodes = []
    for _ in range(group_size):
        episode = run_rollout_with_trajectory(
            leader=leader,
            question=question,
            reference_answer=reference_answer,
            chunks=chunks,
            workers=workers,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
        )
        episodes.append(episode)
    return episodes
