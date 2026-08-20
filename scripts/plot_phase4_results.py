"""Plot Phase 4 GRPO training curves from experiments/results/phase4_train.json."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("experiments/results/phase4_train.json")
OUTPUT_DIR = Path("experiments/results")


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    losses: list[float] = data.get("sample_losses", [])
    epoch_rewards: list[float] = data.get("epoch_mean_reward", [])

    if not losses:
        print("No 'sample_losses' found in result file.", file=sys.stderr)
        return 1

    steps = list(range(len(losses)))
    num_epochs = max(len(epoch_rewards), 1)
    samples_per_epoch = len(losses) // num_epochs

    # Broadcast each epoch's mean reward to its samples for plotting/CSV.
    rewards: list[float] = []
    for reward in epoch_rewards:
        rewards.extend([reward] * samples_per_epoch)
    if rewards:
        # Pad/truncate to match loss length in case of uneven division.
        rewards.extend([epoch_rewards[-1]] * (len(losses) - len(rewards)))
    rewards = rewards[: len(losses)]

    # Save CSV
    csv_path = OUTPUT_DIR / "phase4_training_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "reward"])
        for step, loss, reward in zip(steps, losses, rewards):
            writer.writerow([step, loss, reward])

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Training step (sample index)")
    ax1.set_ylabel("GRPO loss", color=color1)
    ax1.plot(steps, losses, color=color1, alpha=0.8, label="loss")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mean reward", color=color2)
    ax2.plot(steps, rewards, color=color2, alpha=0.4, linestyle="--")
    ax2.scatter(steps, rewards, color=color2, s=12, alpha=0.6, label="reward")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Phase 4 GRPO Training Curves")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    png_path = OUTPUT_DIR / "phase4_training_curves.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
