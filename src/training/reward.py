"""GRPO 训练的标量奖励函数。

奖励设计目标：激励策略以最少资源找到正确答案。
"""
import numpy as np

from src.evaluation.metrics import contains_answer


def compute_reward(
    prediction: str,
    reference: str,
    token_cost: int,
    worker_calls: int,
    accuracy_weight: float = 1.0,
    efficiency_weight: float = 0.1,
    max_token_budget: int = 15000,
    max_worker_budget: int = 25,
) -> float:
    """计算结合准确率与效率的标量奖励。

    奖励公式：
        R = accuracy_weight * is_correct - efficiency_weight * normalized_cost

    其中 normalized_cost = (token_cost / max_token_budget + worker_calls / max_worker_budget) / 2

    奖励值域约为 [-efficiency_weight, +accuracy_weight]：
        - 正确答案 + 零消耗: +accuracy_weight (≈ +1.0)
        - 正确答案 + 满预算: +accuracy_weight - efficiency_weight (≈ +0.9)
        - 错误答案 + 零消耗: 0.0
        - 错误答案 + 满预算: -efficiency_weight (≈ -0.1)

    Args:
        prediction: 模型预测的答案。
        reference: 参考答案（ground truth）。
        token_cost: 本次 episode 消耗的总 token 数。
        worker_calls: Worker 调用次数。
        accuracy_weight: 正确回答的奖励权重，默认 1.0。
        efficiency_weight: 资源消耗的惩罚系数，默认 0.1。
        max_token_budget: token 消耗的归一化常数，默认 15000（约全广播消耗）。
        max_worker_budget: Worker 调用次数的归一化常数，默认 25（约全广播调用数）。

    Returns:
        标量奖励值。
    """
    is_correct = 1.0 if contains_answer(prediction, reference) else 0.0

    # 归一化消耗到 [0, 1] 区间
    token_cost_norm = min(token_cost / max_token_budget, 1.0)
    worker_cost_norm = min(worker_calls / max_worker_budget, 1.0)

    # 两个维度的均值
    normalized_cost = (token_cost_norm + worker_cost_norm) / 2.0

    reward = accuracy_weight * is_correct - efficiency_weight * normalized_cost

    return reward


def compute_group_relative_advantages(rewards: list[float]) -> list[float]:
    """计算 GRPO 所需的组内相对优势。

    给定同一 prompt 的 G 个 rollout 的奖励，计算优势：
        A_i = (R_i - mean(R)) / std(R)

    Args:
        rewards: G 个标量奖励的列表。

    Returns:
        G 个优势值的列表。若所有奖励相同，则全返回 0.0。
    """
    rewards_array = np.array(rewards, dtype=np.float64)
    mean_r = rewards_array.mean()
    std_r = rewards_array.std()

    if std_r < 1e-8:
        # 所有奖励相同，无优势信号
        return [0.0] * len(rewards)

    advantages = ((rewards_array - mean_r) / std_r).tolist()
    return advantages
