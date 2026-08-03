"""Leader 智能体：协调多个 Worker 回答问题。"""
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.agents.worker import NO_MENTION_MARKER
from src.models.model_loader import apply_chat_and_generate


class Leader:
    """模拟 LONGAGENT 中的 Leader，负责拆解问题、综合证据、输出答案。"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_rounds: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_rounds = max_rounds

    def _build_synthesis_prompt(
        self,
        question: str,
        worker_responses: list[tuple[int, str]],
    ) -> list[dict[str, str]]:
        """构建 Leader 综合 Worker 回答的 prompt。"""
        system_prompt = (
            "你是一个协调者，负责根据多个文档阅读者的回答综合出最终答案。"
            "如果阅读者提供的证据足够回答问题，请直接给出最终答案。"
            "如果证据不足或存在矛盾，请明确指出。"
            "回答必须简洁，只包含最终结论。"
        )
        answers_text = "\n\n".join(
            f"[阅读者 {wid}]\n{resp}"
            for wid, resp in worker_responses
            if resp != NO_MENTION_MARKER
        )
        if not answers_text:
            answers_text = "所有阅读者都未提及相关信息。"

        user_prompt = (
            f"问题：{question}\n\n"
            f"各阅读者的回答如下：\n{answers_text}\n\n"
            "请直接给出最终答案。如果无法回答，请回复'无法确定'。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def answer(
        self,
        question: str,
        worker_responses: list[tuple[int, str]],
        max_new_tokens: int = 128,
    ) -> str:
        """根据所有 Worker 的回答综合出最终答案。"""
        messages = self._build_synthesis_prompt(question, worker_responses)
        return apply_chat_and_generate(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        ).strip()

    def run_multi_agent_loop(
        self,
        question: str,
        chunks: list[str],
        workers: list,
    ) -> dict:
        """运行简化的多智能体协作 loop。

        Phase 1 采用最简策略：
        - 第 1 轮：Leader 把原始问题广播给所有 Worker
        - 收集所有 Worker 回答
        - Leader 直接综合答案

        Args:
            question: 用户问题。
            chunks: 文档切分后的 chunk 列表。
            workers: Worker 实例列表（数量应与 chunks 一致）。

        Returns:
            包含最终答案、Worker 回答、调用次数等信息的字典。
        """
        assert len(chunks) == len(workers), "chunks 和 workers 数量必须一致"

        worker_responses = []
        worker_calls = 0
        round_history = []

        # 第 1 轮：广播原始问题
        current_question = question
        for chunk, worker in zip(chunks, workers):
            response = worker.answer(current_question, chunk)
            worker_responses.append((worker.worker_id, response))
            worker_calls += 1

        round_history.append({
            "round": 1,
            "question": current_question,
            "responses": worker_responses.copy(),
        })

        # Leader 综合
        final_answer = self.answer(question, worker_responses)

        return {
            "final_answer": final_answer,
            "worker_responses": worker_responses,
            "worker_calls": worker_calls,
            "rounds": round_history,
        }

