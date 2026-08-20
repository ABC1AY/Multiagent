"""GRPO 训练循环：用 G 个 rollout 的 group-relative advantage 更新 Leader 调度策略。

调用关系：
    scripts/run_phase4_train.py
        └── src.training.grpo_trainer.train_phase4
                ├── model_loader.load_model_and_tokenizer
                ├── peft (LoRA)
                ├── triviaqa_loader.ensure_triviaqa_dataset
                ├── chunking.chunk_document
                ├── rollout.generate_rollout_batch
                └── reward.compute_reward / compute_group_relative_advantages
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.agents.leader import Leader
from src.agents.worker import Worker
from src.data_generation.chunking import chunk_document
from src.data_generation.triviaqa_loader import ensure_triviaqa_dataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.reward import compute_group_relative_advantages, compute_reward
from src.training.rollout import generate_rollout_batch
from src.training.types import Episode
from src.config import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LoRA setup
# ----------------------------------------------------------------------------
def setup_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """把 LoRA 适配器挂到 Qwen 上，让 GRPO 只更新小矩阵。

    默认 target Qwen2 attention + MLP 的所有 linear 层。
    全模型 3B，但可训练参数量通常 < 0.5%。
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "请先装 peft: pip install peft"
        ) from e

    if target_modules is None:
        # Qwen2 的标准 linear 层名
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------
# GRPO 损失
# ----------------------------------------------------------------------------
def compute_grpo_loss(
    episodes: list[Episode],
    advantages: list[float],
    device: torch.device,
) -> torch.Tensor:
    """GRPO 损失 = -mean(A_i * sequence_log_prob_i)。

    原理:
      好的 trajectory (A > 0)  → 提高它的 log_prob（即提高选择该路径的概率）
      差的 trajectory (A < 0)  → 降低它的 log_prob

    sequence_log_prob 来自 Episode.sequence_log_prob（trajectory 内所有 token log-prob 之和）
    """
    assert len(episodes) == len(advantages)
    if not episodes:
        return torch.tensor(0.0, requires_grad=True, device=device)

    adv = torch.tensor(advantages, dtype=torch.float32, device=device)

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    log_probs = torch.stack([_to_tensor(ep.sequence_log_prob) for ep in episodes])

        # GRPO objective: policy gradient with group baseline
    loss = -(adv * log_probs).mean()
    return loss


# ----------------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """GRPO 训练超参集中放这里。"""
    num_samples: int = 20          # 用多少条 TriviaQA
    group_size: int = 4            # 每个 prompt 采 G 个 rollout
    learning_rate: float = 1e-5     # LoRA 常用 1e-5 ~ 5e-5
    num_epochs: int = 3            # 整数据集过几遍
    lora_r: int = 16
    lora_alpha: int = 32
    use_lora: bool = True
    top_k: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    accuracy_weight: float = 1.0
    efficiency_weight: float = 0.1
    log_every_n: int = 1           # 每 N 个 sample 打 log
    max_grad_norm: float = 1.0      # 梯度裁剪，防爆
    checkpoint_every: int = 10     # save LoRA adapter every N samples


def _save_lora_checkpoint(
    model,
    global_step: int,
    mean_reward: float,
    loss: float,
    ckpt_base_dir: Path,
) -> None:
    """Save LoRA adapter and a small metadata file; failures are logged, not raised."""
    ckpt_dir = ckpt_base_dir / f"iter_{global_step}" / "adapter"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        metadata = {"iter": global_step, "mean_reward": float(mean_reward), "loss": float(loss)}
        metadata_path = ckpt_dir.parent / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved LoRA checkpoint to {ckpt_dir}")
    except Exception as exc:
        logger.warning(f"Failed to save checkpoint at iter {global_step}: {exc}")

def train_phase4(cfg: TrainingConfig | None = None) -> dict:
    """Phase 4 主训练循环。返回训练统计信息。"""
    if cfg is None:
        cfg = TrainingConfig()

    logger.info("Loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    if cfg.use_lora:
        logger.info(f"Setting up LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
        model = setup_lora(model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
        # Freeze base model; only LoRA parameters are trainable
        for name, p in model.named_parameters():
            p.requires_grad = "lora_" in name
    else:
        logger.warning("LoRA disabled — full fine-tuning! Will use much more VRAM.")

    # Optimizer 只更新 requires_grad=True 的参数
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate)

    logger.info("Loading TriviaQA dataset...")
    samples = ensure_triviaqa_dataset(use_sample=False)
    samples = samples[: cfg.num_samples]
    logger.info(f"Using {len(samples)} samples for training")
    # checkpoint directory
    ckpt_base_dir = EXPERIMENTS_DIR / "checkpoints" / "phase4"
    ckpt_base_dir.mkdir(parents=True, exist_ok=True)

    # 训练统计
    stats = {
        "sample_losses": [],
        "sample_rewards": [],
        "sample_advantages": [],
        "epoch_mean_reward": [],
    }

    for epoch in range(cfg.num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{cfg.num_epochs} ===")
        epoch_rewards = []

        for sample_idx, sample in enumerate(tqdm(samples, desc=f"epoch {epoch+1}")):
            # 1. 准备 chunks + workers
            chunks = chunk_document(sample["document"], tokenizer, 512, 50)
            workers = [Worker(model, tokenizer, worker_id=i) for i in range(len(chunks))]
            leader = Leader(model, tokenizer)

            # 2. 采样 G 条 trajectory（rollout.py 已经有）
            episodes = generate_rollout_batch(
                leader=leader,
                question=sample["question"],
                reference_answer=sample["answer"],
                chunks=chunks,
                workers=workers,
                group_size=cfg.group_size,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

            # 3. 给每个 episode 算 reward
            for ep in episodes:
                ep.reward = compute_reward(
                    prediction=ep.final_prediction,
                    reference=ep.reference_answer,
                    token_cost=ep.total_token_cost,
                    worker_calls=ep.total_worker_calls,
                    accuracy_weight=cfg.accuracy_weight,
                    efficiency_weight=cfg.efficiency_weight,
                )

            # 4. 算 group-relative advantage
            if not episodes:
                logger.warning("Empty rollout batch, skipping sample.")
                continue

            rewards = [ep.reward for ep in episodes]
            advantages = compute_group_relative_advantages(rewards)

            # 5. GRPO loss + backward + step
            loss = compute_grpo_loss(episodes, advantages, model.device)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # 记录
            stats["sample_losses"].append(loss.item())
            stats["sample_rewards"].append(rewards)
            stats["sample_advantages"].append(advantages)
            epoch_rewards.append(sum(rewards) / len(rewards))

            if len(stats["sample_losses"]) % cfg.log_every_n == 0:
                logger.info(
                    f"  sample {sample['id']}: "
                    f"rewards={[f'{r:.3f}' for r in rewards]}, "
                    f"adv=[{', '.join(f'{a:+.2f}' for a in advantages)}], "
                    f"loss={loss.item():.4f}"
                )
            global_step = epoch * cfg.num_samples + sample_idx + 1
            if cfg.use_lora and global_step % cfg.checkpoint_every == 0:
                _save_lora_checkpoint(
                    model=model,
                    global_step=global_step,
                    mean_reward=epoch_rewards[-1] if epoch_rewards else 0.0,
                    loss=loss.item(),
                    ckpt_base_dir=ckpt_base_dir,
                )

        stats["epoch_mean_reward"].append(sum(epoch_rewards) / max(len(epoch_rewards), 1))
        logger.info(
            f"Epoch {epoch+1} done. Mean reward: {stats['epoch_mean_reward'][-1]:.4f}"
        )

        if cfg.use_lora and stats["sample_losses"]:
            _save_lora_checkpoint(
                model=model,
                global_step=(epoch + 1) * cfg.num_samples,
                mean_reward=stats["epoch_mean_reward"][-1],
                loss=stats["sample_losses"][-1],
                ckpt_base_dir=ckpt_base_dir,
            )

    return stats
