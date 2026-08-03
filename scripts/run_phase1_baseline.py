"""Phase 1：简化版 LONGAGENT 基线实验。"""
import json
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# 把项目根目录加入 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.agents.leader import Leader
from src.agents.worker import Worker
from src.baselines import (
    count_prompt_tokens,
    single_model_full_context,
    single_model_truncated,
)
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.data_generation.chunking import chunk_document, count_tokens
from src.data_generation.needle_in_haystack import generate_dataset
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.model_loader import load_model_and_tokenizer

# 实验参数
NUM_SAMPLES = 20  # 第一次跑建议用 20 条快速验证
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_NEW_TOKENS = 64
TRUNCATED_MAX_TOKENS = 2048
RANDOM_SEED = 42


def ensure_dataset(path: Path, num_samples: int) -> list[dict]:
    """如果数据集不存在则生成。"""
    if path.exists():
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Loaded {len(samples)} samples from {path}")
        return samples
    return generate_dataset(num_samples=num_samples, output_path=path)


def run_multi_agent(
    model,
    tokenizer,
    samples: list[dict],
) -> tuple[list[str], list[int], list[int]]:
    """运行多智能体基线。"""
    leader = Leader(model, tokenizer)
    predictions = []
    token_counts = []
    worker_calls_list = []

    for sample in tqdm(samples, desc="Multi-Agent"):
        chunks = chunk_document(
            sample["document"], tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
        )
        workers = [Worker(model, tokenizer, worker_id=i) for i in range(len(chunks))]

        result = leader.run_multi_agent_loop(sample["question"], chunks, workers)
        predictions.append(result["final_answer"])
        worker_calls_list.append(result["worker_calls"])

        # 统计 token：所有 prompt 中输入 token 的总和（近似）
        total_input_tokens = 0
        for wid, resp in result["worker_responses"]:
            chunk = chunks[wid]
            prompt = (
                f"你是一个文档阅读者。问题：{sample['question']} 文档片段：{chunk}"
            )
            total_input_tokens += count_tokens(prompt, tokenizer)
        # Leader 综合 prompt
        answers_text = "\n".join(
            f"[阅读者 {wid}] {resp}"
            for wid, resp in result["worker_responses"]
        )
        leader_prompt = f"问题：{sample['question']} 各阅读者回答：{answers_text}"
        total_input_tokens += count_tokens(leader_prompt, tokenizer)
        # 输出 token
        total_input_tokens += count_tokens(result["final_answer"], tokenizer)
        token_counts.append(total_input_tokens)

    return predictions, token_counts, worker_calls_list


def run_baseline_full(
    model,
    tokenizer,
    samples: list[dict],
) -> tuple[list[str], list[int]]:
    """运行单模型全文基线。"""
    predictions = []
    token_counts = []
    for sample in tqdm(samples, desc="Full-Context Baseline"):
        pred = single_model_full_context(
            model, tokenizer, sample["question"], sample["document"], MAX_NEW_TOKENS
        )
        predictions.append(pred)
        token_counts.append(
            count_prompt_tokens(sample["document"], sample["question"], tokenizer)
            + count_tokens(pred, tokenizer)
        )
    return predictions, token_counts


def run_baseline_truncated(
    model,
    tokenizer,
    samples: list[dict],
) -> tuple[list[str], list[int]]:
    """运行单模型截断基线。"""
    predictions = []
    token_counts = []
    for sample in tqdm(samples, desc="Truncated Baseline"):
        pred = single_model_truncated(
            model,
            tokenizer,
            sample["question"],
            sample["document"],
            TRUNCATED_MAX_TOKENS,
            MAX_NEW_TOKENS,
        )
        predictions.append(pred)
        token_counts.append(TRUNCATED_MAX_TOKENS + count_tokens(pred, tokenizer))
    return predictions, token_counts


def main():
    random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_PROCESSED_DIR / "needle_in_haystack.jsonl"
    samples = ensure_dataset(dataset_path, NUM_SAMPLES)
    samples = samples[:NUM_SAMPLES]

    print(f"\nRunning Phase 1 baseline on {len(samples)} samples...")
    print(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")

    # 加载模型（单实例，Leader 和 Worker 共享）
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    references = [s["answer"] for s in samples]

    # 1. 多智能体基线
    start = time.time()
    ma_preds, ma_tokens, ma_calls = run_multi_agent(model, tokenizer, samples)
    ma_time = time.time() - start
    ma_metrics = compute_metrics(ma_preds, references, tokenizer, ma_tokens, ma_calls)
    print_metrics("Multi-Agent (LONGAGENT-style)", ma_metrics)
    print(f"Total time: {ma_time:.1f}s")

    # 清显存，避免 baseline 受 multi-agent 碎片影响
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 单模型全文基线
    start = time.time()
    full_preds, full_tokens = run_baseline_full(model, tokenizer, samples)
    full_time = time.time() - start
    full_metrics = compute_metrics(full_preds, references, tokenizer, full_tokens)
    print_metrics("Single-Model Full Context", full_metrics)
    print(f"Total time: {full_time:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. 单模型截断基线
    start = time.time()
    trunc_preds, trunc_tokens = run_baseline_truncated(model, tokenizer, samples)
    trunc_time = time.time() - start
    trunc_metrics = compute_metrics(trunc_preds, references, tokenizer, trunc_tokens)
    print_metrics("Single-Model Truncated", trunc_metrics)
    print(f"Total time: {trunc_time:.1f}s")

    # 保存结果
    result = {
        "num_samples": len(samples),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "multi_agent": {
            "metrics": ma_metrics,
            "time": ma_time,
            "predictions": ma_preds,
        },
        "full_context": {
            "metrics": full_metrics,
            "time": full_time,
            "predictions": full_preds,
        },
        "truncated": {
            "metrics": trunc_metrics,
            "time": trunc_time,
            "predictions": trunc_preds,
        },
        "references": references,
    }

    output_path = RESULTS_DIR / "phase1_baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

