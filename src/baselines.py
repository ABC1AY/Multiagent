"""单模型基线：长上下文直接读取 vs. 截断读取。"""
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data_generation.chunking import count_tokens
from src.models.model_loader import apply_chat_and_generate


def _safe_generate(generate_fn, oom_fallback: str = "[OOM: input too long for GPU]"):
    """包装生成函数，捕获 CUDA OOM 并返回占位答案。"""
    try:
        return generate_fn()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return oom_fallback


def single_model_full_context(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    document: str,
    max_new_tokens: int = 128,
) -> str:
    """把完整文档塞入单模型 prompt。"""
    system_prompt = "你是一位阅读助手，请根据下面提供的完整文档回答问题。"
    user_prompt = f"文档：\n{document}\n\n问题：{question}\n\n请给出简短回答："
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _safe_generate(
        lambda: apply_chat_and_generate(
            model, tokenizer, messages, max_new_tokens=max_new_tokens, do_sample=False
        ).strip()
    )


def single_model_truncated(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    document: str,
    max_context_tokens: int = 2048,
    max_new_tokens: int = 128,
) -> str:
    """只把文档前 max_context_tokens 个 token 塞入单模型 prompt。"""
    tokens = tokenizer.encode(document, add_special_tokens=False)
    truncated = tokenizer.decode(tokens[:max_context_tokens], skip_special_tokens=True)
    system_prompt = "你是一位阅读助手，请根据下面提供的文档片段回答问题。"
    user_prompt = f"文档片段：\n{truncated}\n\n问题：{question}\n\n请给出简短回答："
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _safe_generate(
        lambda: apply_chat_and_generate(
            model, tokenizer, messages, max_new_tokens=max_new_tokens, do_sample=False
        ).strip()
    )


def count_prompt_tokens(document: str, question: str, tokenizer: PreTrainedTokenizer) -> int:
    """统计单模型基线 prompt 的 token 数。"""
    prompt = f"文档：\n{document}\n\n问题：{question}\n\n请给出简短回答："
    return count_tokens(prompt, tokenizer)

