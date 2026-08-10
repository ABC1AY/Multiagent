"""GRPO 训练模块。

提供训练所需的核心组件：
- ``reward``: 标量奖励函数
- ``types``: 轨迹数据结构
- ``rollout``: Episode 轨迹记录器
"""
from src.training.reward import compute_group_relative_advantages, compute_reward
from src.training.rollout import (
    RolloutRecorder,
    generate_rollout_batch,
    run_rollout_with_trajectory,
)
from src.training.types import (
    DecisionPoint,
    Episode,
    LeaderAction,
    WorkerAction,
)

__all__ = [
    "compute_reward",
    "compute_group_relative_advantages",
    "RolloutRecorder",
    "run_rollout_with_trajectory",
    "generate_rollout_batch",
    "Episode",
    "DecisionPoint",
    "WorkerAction",
    "LeaderAction",
]
