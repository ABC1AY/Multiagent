"""Phase 4 训练入口。

用法:
    python scripts/run_phase4_train.py
    python scripts/run_phase4_train.py --num-samples 50 --epochs 3
    python scripts/run_phase4_train.py --no-lora       # 跑全参微调（不推荐）
"""
import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.training.grpo_trainer import TrainingConfig, train_phase4
from src.config import EXPERIMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 GRPO training for SwarmAgent")
    p.add_argument("--num-samples", type=int, default=20,
                   help="用多少条 TriviaQA 训练（默认 20）")
    p.add_argument("--epochs", type=int, default=3,
                   help="epoch 数（默认 3）")
    p.add_argument("--group-size", type=int, default=4,
                   help="GRPO 每组 G 个 rollout（默认 4）")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="learning rate（默认 1e-5）")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank（默认 16）")
    p.add_argument("--no-lora", action="store_true",
                   help="不用 LoRA，全参微调（显存不够别用）")
    p.add_argument("--output", type=str,
                   default=str(EXPERIMENTS_DIR / "phase4_train.json"),
                   help="训练结果保存路径")
    p.add_argument("--accuracy-weight", type=float, default=1.0,
                   help="reward accuracy weight")
    p.add_argument("--efficiency-weight", type=float, default=0.1,
                   help="reward efficiency weight")
    p.add_argument("--top-k", type=int, default=5,
                   help="BM25 seed retrieval top-k")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus sampling top-p")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="gradient clipping max norm")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="save LoRA adapter every N samples")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainingConfig(
        num_samples=args.num_samples,
        group_size=args.group_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        use_lora=not args.no_lora,
        accuracy_weight=args.accuracy_weight,
        efficiency_weight=args.efficiency_weight,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_grad_norm=args.max_grad_norm,
        checkpoint_every=args.checkpoint_every,
    )

    logger.info("=" * 60)
    logger.info("Phase 4 GRPO Training")
    logger.info("=" * 60)
    logger.info(f"  num_samples={cfg.num_samples}")
    logger.info(f"  epochs={cfg.num_epochs}")
    logger.info(f"  group_size={cfg.group_size}")
    logger.info(f"  learning_rate={cfg.learning_rate}")
    logger.info(f"  use_lora={cfg.use_lora} (r={cfg.lora_r})")
    logger.info(f"  accuracy_weight={cfg.accuracy_weight}")
    logger.info(f"  efficiency_weight={cfg.efficiency_weight}")
    logger.info(f"  top_k={cfg.top_k}")
    logger.info(f"  temperature={cfg.temperature}")
    logger.info(f"  top_p={cfg.top_p}")
    logger.info(f"  max_grad_norm={cfg.max_grad_norm}")
    logger.info(f"  checkpoint_every={cfg.checkpoint_every}")
    logger.info("=" * 60)

    stats = train_phase4(cfg)

    # 保存训练结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg.__dict__,
                "epoch_mean_reward": stats["epoch_mean_reward"],
                "n_samples": len(stats["sample_losses"]),
                "mean_loss_final_epoch": (
                    sum(stats["sample_losses"][-cfg.num_samples:]) / cfg.num_samples
                    if stats["sample_losses"] else None
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"训练结果已保存到: {out_path}")
    logger.info(f"每 epoch 平均 reward: {stats['epoch_mean_reward']}")


if __name__ == "__main__":
    main()
