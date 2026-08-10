"""GRPO 训练轨迹的数据结构定义。"""
from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class WorkerAction:
    """单次 Worker 查询动作。"""

    worker_id: int
    chunk_index: int
    question: str
    chunk_text: str
    response_text: str
    response_token_ids: list[int]
    response_log_probs: list[float]


@dataclass
class LeaderAction:
    """Leader 的综合/决策动作。"""

    action_type: Literal["synthesis", "conflict_resolution", "stop"]
    worker_responses_used: list[int]  # 使用的 worker ID 列表
    generated_text: str
    generated_token_ids: list[int]
    generated_log_probs: list[float]


@dataclass
class DecisionPoint:
    """Episode 中的单个决策点。"""

    step: int
    state: dict  # 序列化状态：question, retrieved chunks, accumulated responses
    action: Union[WorkerAction, LeaderAction]
    action_type: Literal[
        "worker_query", "leader_synthesis", "conflict_resolution", "stop"
    ]


@dataclass
class Episode:
    """完整的 episode 轨迹。

    包含一次 rollout 中的所有决策点、最终预测、资源消耗和奖励。
    """

    episode_id: str
    question: str
    reference_answer: str
    document_chunks: list[str]

    # 轨迹
    decisions: list[DecisionPoint] = field(default_factory=list)

    # 结果
    final_prediction: str = ""
    total_token_cost: int = 0
    total_worker_calls: int = 0
    reward: float = 0.0

    # 元数据
    metadata: dict = field(default_factory=dict)

    @property
    def sequence_log_prob(self) -> float:
        """计算整个 episode 的序列对数概率（所有动作 log_prob 之和）。"""
        total = 0.0
        for decision in self.decisions:
            action = decision.action
            if hasattr(action, "response_log_probs"):
                total += sum(action.response_log_probs)
            elif hasattr(action, "generated_log_probs"):
                total += sum(action.generated_log_probs)
        return total
