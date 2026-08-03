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

    def __repr__(self):
        return f"Worker(id={self.worker_id})"

