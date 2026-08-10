"""Phase 3：多轮 Agent 调度实验。

对比：
- Phase 2 selective_full（硬编码召回 + 冲突消解 + fallback）
- Phase 3 agent_loop（模型自己决定 QUERY / VERIFY / ANSWER）
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

NUM_SAMPLES = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_NEW_TOKENS = 64
RANDOM_SEED = 42

SEED_TOP_K = 5
AGENT_MAX_ROUNDS = 4
AGENT_BUDGET = 10


def ensure_dataset(path: Path, num_samples: int) -> list[dict]:
    return ensure_triviaqa_dataset(output_path=path, use_sample=False)


def _count_tokens_in_result(
    question: str,
    chunks: list[str],
    result: dict,
    tokenizer,
) -> int:
    """估算一次实验消耗的 token。"""
    total = 0
    for round_info in result.get("rounds", []):
        rtype = round_info.get("type", "")
        if rtype in ("selective_query", "seed_selective_query"):
            for wid, _ in round_info["responses"]:
                prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[wid]}"
                total += count_tokens(prompt, tokenizer)
        elif rtype == "conflict_resolution":
            total += 3 * count_tokens(f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[0]}", tokenizer)
        elif rtype in ("broadcast", "fallback_broadcast"):
            for chunk in chunks:
                prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunk}"
                total += count_tokens(prompt, tokenizer)
        elif rtype == "agent_action":
            action = round_info.get("action", "")
            if action.startswith("QUERY["):
                for wid, _ in round_info["responses"]:
                    prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[wid]}"
                    total += count_tokens(prompt, tokenizer)
            elif action == "QUERY_ALL":
                for chunk in chunks:
                    prompt = f"你是一个文档阅读者。问题：{question} 文档片段：{chunk}"
                    total += count_tokens(prompt, tokenizer)
            elif action.startswith("VERIFY"):
                total += 3 * count_tokens(f"你是一个文档阅读者。问题：{question} 文档片段：{chunks[0]}", tokenizer)
            # Agent 决策 prompt 近似
            total += 200

    # Leader 综合 prompt
    answers_text = "\n".join(
        f"[阅读者{wid}] {resp}" for wid, resp in result["worker_responses"]
    )
    leader_prompt = f"问题：{question} 各阅读者回答：{answers_text}"
    total += count_tokens(leader_prompt, tokenizer)
    total += count_tokens(result["final_answer"], tokenizer)
    return total


def run_agent_method(
    model,
    tokenizer,
    samples: list[dict],
) -> tuple[list[str], list[int], list[int], list[float], list[str], list[list[dict]]]:
    leader = Leader(model, tokenizer)
    predictions = []
    token_counts = []
    worker_calls_list = []
    confidence_list = []
    stop_reasons = []
    histories = []

    for sample in tqdm(samples, desc="Phase3 Agent"):
        chunks = chunk_document(
            sample["document"], tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
        )
        workers = [Worker(model, tokenizer, worker_id=i) for i in range(len(chunks))]

        result = leader.run_agent_loop(
            sample["question"],
            chunks,
            workers,
            seed_top_k=SEED_TOP_K,
            max_rounds=AGENT_MAX_ROUNDS,
            budget=AGENT_BUDGET,
        )

        predictions.append(result["final_answer"])
        worker_calls_list.append(result["worker_calls"])
        confidence_list.append(result.get("confidence", 0.0))
        stop_reasons.append(result.get("stop_reason", ""))
        histories.append(result.get("rounds", []))
        token_counts.append(
            _count_tokens_in_result(sample["question"], chunks, result, tokenizer)
        )

    return predictions, token_counts, worker_calls_list, confidence_list, stop_reasons, histories


def run_selective_full(
    model,
    tokenizer,
    samples: list[dict],
) -> tuple[list[str], list[int], list[int], list[float]]:
    leader = Leader(model, tokenizer)
    predictions = []
    token_counts = []
    worker_calls_list = []
    confidence_list = []

    for sample in tqdm(samples, desc="Phase2 selective_full"):
        chunks = chunk_document(
            sample["document"], tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
        )
        workers = [Worker(model, tokenizer, worker_id=i) for i in range(len(chunks))]
        top_k = min(5, len(chunks))
        result = leader.run_selective_loop(
            sample["question"], chunks, workers, top_k=top_k,
            resolve_conflict=True, allow_fallback=True, confidence_threshold=1.0
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

    print(f"\nRunning Phase 3 experiments on {len(samples)} samples...")
    print(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"Agent seed_top_k={SEED_TOP_K}, max_rounds={AGENT_MAX_ROUNDS}, budget={AGENT_BUDGET}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    references = [s["answer"] for s in samples]
    results = {}

    # Phase 2 selective_full baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    start = time.time()
    preds, tokens, calls, confs = run_selective_full(model, tokenizer, samples)
    elapsed = time.time() - start
    metrics = compute_metrics(preds, references, tokenizer, tokens, calls)
    metrics["avg_confidence"] = sum(confs) / len(confs) if confs else 0.0
    print_metrics("selective_full", metrics)
    print(f"Total time: {elapsed:.1f}s")
    results["selective_full"] = {
        "metrics": metrics,
        "time": elapsed,
        "predictions": preds,
        "confidence": confs,
    }

    # Phase 3 agent_loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    start = time.time()
    preds, tokens, calls, confs, stops, histories = run_agent_method(model, tokenizer, samples)
    elapsed = time.time() - start
    metrics = compute_metrics(preds, references, tokenizer, tokens, calls)
    metrics["avg_confidence"] = sum(confs) / len(confs) if confs else 0.0
    print_metrics("agent_loop", metrics)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Stop reasons: {stops}")
    results["agent_loop"] = {
        "metrics": metrics,
        "time": elapsed,
        "predictions": preds,
        "confidence": confs,
        "stop_reasons": stops,
        "round_histories": histories,
    }

    output_path = RESULTS_DIR / "phase3_agent.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
