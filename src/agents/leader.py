"""Leader 智能体：协调多个 Worker 回答问题。"""
import re
import string
from collections import Counter

from transformers import PreTrainedModel, PreTrainedTokenizer

from src.agents.worker import NO_MENTION_MARKER
from src.models.model_loader import apply_chat_and_generate
from src.retrieval.bm25_retriever import BM25Retriever


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

    def _normalize_answer(self, text: str) -> str:
        """归一化答案文本，用于判断两个回答是否一致。"""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", "", text)
        return text

    def _group_mentions(
        self,
        worker_responses: list[tuple[int, str]],
    ) -> dict[str, list[int]]:
        """把非'未提及'的回答按归一化后文本分组。"""
        groups: dict[str, list[int]] = {}
        for wid, resp in worker_responses:
            if resp == NO_MENTION_MARKER:
                continue
            key = self._normalize_answer(resp)
            groups.setdefault(key, []).append(wid)
        return groups

    def _compute_confidence(
        self,
        worker_responses: list[tuple[int, str]],
    ) -> float:
        """计算当前证据的置信度。

        规则：在返回了具体答案的 Worker 中，最大一致组的比例。
        例如 5 个 Worker 查询，1 个返回答案、4 个未提及 -> 置信度 1.0。
        """
        groups = self._group_mentions(worker_responses)
        if not groups:
            return 0.0
        max_group_size = max(len(wids) for wids in groups.values())
        total_mentions = sum(len(wids) for wids in groups.values())
        return max_group_size / total_mentions

    def _detect_conflict(
        self,
        worker_responses: list[tuple[int, str]],
    ) -> bool:
        """判断当前回答中是否存在冲突（两个不同的具体答案）。"""
        groups = self._group_mentions(worker_responses)
        return len(groups) >= 2

    def _selective_query(
        self,
        question: str,
        chunks: list[str],
        workers: list,
        top_k: int,
    ) -> list[tuple[int, str]]:
        """先召回 top-K chunk，再只查询对应的 Worker。"""
        retriever = BM25Retriever(self.tokenizer).fit(chunks)
        top_results = retriever.retrieve(question, top_k=top_k)
        top_indices = [idx for idx, _ in top_results]

        responses = []
        for idx in top_indices:
            worker = workers[idx]
            response = worker.answer(question, chunks[idx])
            responses.append((worker.worker_id, response))
        return responses

    def _resolve_conflict(
        self,
        question: str,
        worker_responses: list[tuple[int, str]],
        chunks: list[str],
        workers: list,
    ) -> list[tuple[int, str]]:
        """冲突消解：让冲突的 Worker 互换 chunk 重读，并加入仲裁 Worker。"""
        groups = self._group_mentions(worker_responses)
        if len(groups) < 2:
            return worker_responses

        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        conflict_wids = sorted_groups[0][1][:1] + sorted_groups[1][1][:1]

        new_responses = list(worker_responses)
        if len(conflict_wids) == 2:
            wid_a, wid_b = conflict_wids
            chunk_a, chunk_b = chunks[wid_a], chunks[wid_b]
            resp_a = workers[wid_a].answer(question, chunk_b)
            resp_b = workers[wid_b].answer(question, chunk_a)
            new_responses[wid_a] = (wid_a, resp_a)
            new_responses[wid_b] = (wid_b, resp_b)

        arbitrator_wid = None
        for i, _ in enumerate(workers):
            if i not in conflict_wids:
                arbitrator_wid = i
                break
        if arbitrator_wid is not None:
            combined_chunk = chunks[conflict_wids[0]] + "\n\n" + chunks[conflict_wids[1]]
            arb_resp = workers[arbitrator_wid].answer(question, combined_chunk)
            new_responses.append((arbitrator_wid, f"[仲裁] {arb_resp}"))

        return new_responses

    def run_selective_loop(
        self,
        question: str,
        chunks: list[str],
        workers: list,
        top_k: int = 5,
        resolve_conflict: bool = True,
        allow_fallback: bool = True,
        confidence_threshold: float = 1.0,
    ) -> dict:
        """Phase 2 的选择性调度 loop。"""
        assert len(chunks) == len(workers), "chunks 和 workers 数量必须一致"
        assert top_k <= len(chunks), "top_k 不能超过 chunk 总数"

        worker_calls = 0
        round_history = []

        responses = self._selective_query(question, chunks, workers, top_k)
        worker_calls += top_k
        round_history.append({
            "round": 1,
            "type": "selective_query",
            "question": question,
            "responses": list(responses),
            "retrieved_chunks": [r[0] for r in responses],
        })

        confidence = self._compute_confidence(responses)
        has_conflict = self._detect_conflict(responses)

        if resolve_conflict and has_conflict:
            responses = self._resolve_conflict(question, responses, chunks, workers)
            worker_calls += 3
            round_history.append({
                "round": 2,
                "type": "conflict_resolution",
                "question": question,
                "responses": list(responses),
            })
            confidence = self._compute_confidence(responses)

        if allow_fallback and confidence < confidence_threshold:
            fallback_responses = []
            for chunk, worker in zip(chunks, workers):
                response = worker.answer(question, chunk)
                fallback_responses.append((worker.worker_id, response))
            worker_calls += len(chunks)
            responses = fallback_responses
            round_history.append({
                "round": len(round_history) + 1,
                "type": "fallback_broadcast",
                "question": question,
                "responses": list(responses),
            })
            confidence = self._compute_confidence(responses)

        final_answer = self.answer(question, responses)

        return {
            "final_answer": final_answer,
            "worker_responses": responses,
            "worker_calls": worker_calls,
            "confidence": confidence,
            "rounds": round_history,
        }

    def run_multi_agent_loop(
        self,
        question: str,
        chunks: list[str],
        workers: list,
    ) -> dict:
        """运行简化的多智能体协作 loop（Phase 1 基线）。"""
        assert len(chunks) == len(workers), "chunks 和 workers 数量必须一致"

        worker_responses = []
        worker_calls = 0
        round_history = []

        current_question = question
        for chunk, worker in zip(chunks, workers):
            response = worker.answer(current_question, chunk)
            worker_responses.append((worker.worker_id, response))
            worker_calls += 1

        round_history.append({
            "round": 1,
            "type": "broadcast",
            "question": current_question,
            "responses": worker_responses.copy(),
        })

        final_answer = self.answer(question, worker_responses)

        return {
            "final_answer": final_answer,
            "worker_responses": worker_responses,
            "worker_calls": worker_calls,
            "rounds": round_history,
        }
