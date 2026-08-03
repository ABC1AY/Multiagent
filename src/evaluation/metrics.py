"""评估指标。"""
import re
import string

from transformers import PreTrainedTokenizer

from src.data_generation.chunking import count_tokens


def normalize_text(text: str) -> str:
    """对文本进行归一化，用于答案匹配。"""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 去除多余空白
    text = re.sub(r"\s+", "", text)
    return text


def contains_answer(prediction: str, answer: str) -> bool:
    """判断预测中是否包含参考答案。

    采用宽松的包含匹配，适用于简短事实性答案。
    """
    pred_norm = normalize_text(prediction)
    ans_norm = normalize_text(answer)
    if not ans_norm:
        return False
    return ans_norm in pred_norm


def compute_metrics(
    predictions: list[str],
    references: list[str],
    tokenizer: PreTrainedTokenizer | None = None,
    token_counts: list[int] | None = None,
    worker_calls: list[int] | None = None,
) -> dict:
    """计算准确率、token 消耗、调用次数等指标。"""
    correct = sum(
        contains_answer(pred, ref) for pred, ref in zip(predictions, references)
    )
    total = len(predictions)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }

    if token_counts:
        metrics["total_tokens"] = sum(token_counts)
        metrics["avg_tokens"] = sum(token_counts) / len(token_counts)

    if worker_calls:
        metrics["total_worker_calls"] = sum(worker_calls)
        metrics["avg_worker_calls"] = sum(worker_calls) / len(worker_calls)

    return metrics


def print_metrics(name: str, metrics: dict):
    """打印指标。"""
    print(f"\n===== {name} =====")
    print(f"Total samples : {metrics['total']}")
    print(f"Correct       : {metrics['correct']}")
    print(f"Accuracy      : {metrics['accuracy']:.2%}")
    if "avg_tokens" in metrics:
        print(f"Avg tokens    : {metrics['avg_tokens']:.1f}")
    if "avg_worker_calls" in metrics:
        print(f"Avg worker calls: {metrics['avg_worker_calls']:.1f}")

