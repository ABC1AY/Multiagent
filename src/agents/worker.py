"""Worker 智能体：只读取一个文档 chunk 并回答问题。"""
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.model_loader import apply_chat_and_generate

NO_MENTION_MARKER = "未提及"


class Worker:
    """模拟 LONGAGENT 中的 Member，只访问一个 chunk。"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        worker_id: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.worker_id = worker_id

    def answer(self, question: str, chunk: str, max_new_tokens: int = 64) -> str:
        """根据 chunk 回答问题，找不到时返回 NO_MENTION_MARKER。"""
        system_prompt = (
            "你是一个文档阅读者。你只能看到文档的一个片段。"
            "请仅根据该片段中的信息回答问题。"
            "如果片段中没有相关信息，请只回复'未提及'，不要编造。"
        )
        user_prompt = f"问题：{question}\n\n文档片段：\n{chunk}\n\n请给出简短回答："
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = apply_chat_and_generate(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return response.strip()

    def answer_with_logprobs(
        self,
        question: str,
        chunk: str,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """使用采样回答问题并返回文本、token ID 和对数概率。

        与 ``answer()`` 不同，本方法默认使用采样（``do_sample=True``）
        并返回完整的生成信息，用于 GRPO 训练 rollout。

        Args:
            question: 问题文本。
            chunk: 文档片段。
            max_new_tokens: 最多生成的新 token 数。
            do_sample: 是否使用采样，默认为 True。
            temperature: 采样温度。
            top_p: nucleus 采样的概率阈值。

        Returns:
            包含以下键的字典：
            - ``text``: 生成的文本（已去除首尾空白）
            - ``token_ids``: 生成 token 的 ID 列表
            - ``log_probs``: 每个 token 的对数概率列表
        """
        from src.models.model_loader import generate_with_logprobs

        system_prompt = (
            "你是一个文档阅读者。你只能看到文档的一个片段。"
            "请仅根据该片段中的信息回答问题。"
            "如果片段中没有相关信息，请只回复'未提及'，不要编造。"
        )
        user_prompt = f"问题：{question}\n\n文档片段：\n{chunk}\n\n请给出简短回答："
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text, token_ids, log_probs = generate_with_logprobs(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        return {
            "text": text.strip(),
            "token_ids": token_ids,
            "log_probs": log_probs,
        }

    def __repr__(self):
        return f"Worker(id={self.worker_id})"

