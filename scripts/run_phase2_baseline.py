"""Phase 2：硬编码调度优化实验。

对比：
- Phase 1 全广播基线
- Phase 2 选择性查询（BM25 top-K）
- Phase 2 + 冲突消解
- Phase 2 + 冲突消解 + 停止规则
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.agents.leader import Leader
from src.agents.worker import Worker
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, TRIVIAQA_PROCESSED_PATH
from src.data_generation.chunking import chunk_document, count_tokens
from src.data_generation.triviaqa_loader import ensure_triviaqa_dataset
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.model_loader import load_model_and_tokenizer

NUM_SAMPLES = 20
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_NEW_TOKENS = 64
TOP_K = 5
CONFIDENCE_THRESHOLD = 1.0
RANDOM_SEED = 42
USE_SAMPLE = False  # Set to True to use 5-sample test dataset (when network is unavailable)


def ensure_dataset(path: Path, num_samples: int) -> list[dict]:
    """如果数据集不存在则生成。"""
    return ensure_triviaqa_dataset(output_path=path, use_sample=USE_SAMPLE)


def _count_tokens_in_result(
    question: str,
    chunks: list[str],
    result: dict,
    tokenizer,
) -> int:
    """估算 Phase 2 一次实验消耗的总 token。"""
    total = 0
    for round_info in result.get("rounds", []):
        if round_info["type"] == "selective_query":
            for wid, _ in round_info["responses"]:
                prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[wid]}"
                total += count_tokens(prompt, tokenizer)
        elif round_info["type"] == "conflict_resolution":
            # 近似：互换重读 + 仲裁
            total += 3 * count_tokens(f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[0]}", tokenizer)
        elif round_info["type"] in ("broadcast", "fallback_broadcast"):
            for chunk in chunks:
                prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunk}"
                total += count_tokens(prompt, tokenizer)
    # Leader 综合 prompt
    answers_text = "\n".join(
        f"[阅读者 {wid}] {resp}" for wid, resp in result["worker_responses"]
    )
    leader_prompt = f"问题：{question} 各阅读者回答：{answers_text}"
    total += count_tokens(leader_prompt, tokenizer)
    total += count_tokens(result["final_answer"], tokenizer)
    return total


def run_method(
    model,
    tokenizer,
    samples: list[dict],
    method: str,
) -> tuple[list[str], list[int], list[int], list[float]]:
    """运行指定方法。

    method 可选：
    - 'broadcast': Phase 1 全广播
    - 'selective': Phase 2 选择性查询
    - 'selective_conflict': Phase 2 + 冲突消解
    - 'selective_full': Phase 2 + 冲突消解 + fallback 停止规则
    """
    leader = Leader(model, tokenizer)
    predictions = []
    token_counts = []
    worker_calls_list = []
    confidence_list = []

    desc = {
        "broadcast": "Phase1 Broadcast",
        "selective": "Phase2 Selective",
        "selective_conflict": "Phase2 + Conflict",
        "selective_full": "Phase2 + Conflict + Fallback",
    }[method]

    for sample in tqdm(samples, desc=desc):
        chunks = chunk_document(
            sample["document"], tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
        )
        workers = [Worker(model, tokenizer, worker_id=i) for i in range(len(chunks))]
        top_k = min(TOP_K, len(chunks))

        if method == "broadcast":
            result = leader.run_multi_agent_loop(sample["question"], chunks, workers)
        elif method == "selective":
            result = leader.run_selective_loop(
                sample["question"], chunks, workers, top_k=top_k,
                resolve_conflict=False, allow_fallback=False, confidence_threshold=CONFIDENCE_THRESHOLD
            )
        elif method == "selective_conflict":
            result = leader.run_selective_loop(
                sample["question"], chunks, workers, top_k=top_k,
                resolve_conflict=True, allow_fallback=False, confidence_threshold=CONFIDENCE_THRESHOLD
            )
        else:  # selective_full
            result = leader.run_selective_loop(
                sample["question"], chunks, workers, top_k=top_k,
                resolve_conflict=True, allow_fallback=True, confidence_threshold=CONFIDENCE_THRESHOLD
            )
        predictions.append(result["final_answer"])
        worker_calls_list.append(result["worker_calls"])
        confidence_list.append(result.get("confidence", 0.0))
        token_counts.append(
            _count_tokens_in_result(sample["question"], chunks, result, tokenizer)
        )

    return predictions, token_counts, worker_calls_list, confidence_list


def main():
    random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = TRIVIAQA_PROCESSED_PATH
    samples = ensure_dataset(dataset_path, NUM_SAMPLES)
    samples = samples[:NUM_SAMPLES]

    print(f"\nRunning Phase 2 experiments on {len(samples)} samples...")
    print(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}, top_k: {TOP_K}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    references = [s["answer"] for s in samples]
    results = {}

    methods = ["broadcast", "selective", "selective_conflict", "selective_full"]
    for method in methods:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.time()
        preds, tokens, calls, confs = run_method(model, tokenizer, samples, method)
        elapsed = time.time() - start
        metrics = compute_metrics(preds, references, tokenizer, tokens, calls)
        metrics["avg_confidence"] = sum(confs) / len(confs) if confs else 0.0
        print_metrics(f"{method}", metrics)
        print(f"Total time: {elapsed:.1f}s")
        results[method] = {
            "metrics": metrics,
            "time": elapsed,
            "predictions": preds,
            "confidence": confs,
        }

    output_path = RESULTS_DIR / "phase2_baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

