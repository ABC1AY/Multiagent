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
            # Replace in-place by position, not by worker_id
            for i, (wid, _) in enumerate(new_responses):
                if wid == wid_a:
                    new_responses[i] = (wid_a, resp_a)
                elif wid == wid_b:
                    new_responses[i] = (wid_b, resp_b)

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
    def _build_agent_prompt(
        self,
        question: str,
        chunks: list[str],
        responses: list[tuple[int, str]],
        budget_left: int,
        round_num: int,
        max_rounds: int,
    ) -> list[dict[str, str]]:
        """构建多轮调度 Agent 的 prompt。"""
        system_prompt = (
            "You are a scheduling agent. Your job is to decide the next action to answer a question using document readers (Workers).\n"
            "You start with some evidence from BM25 retrieval. In each round, choose exactly ONE action.\n"
            "Prefer ANSWER as soon as the evidence is sufficient and consistent. Only QUERY or VERIFY when evidence is insufficient or conflicting.\n"
            "Do not query a Worker that has already been queried unless you are performing VERIFY."
        )

        evidence_lines = []
        for wid, resp in responses:
            evidence_lines.append(f"  Worker {wid}: {resp}")
        if not evidence_lines:
            evidence_lines.append("  暂无")
        evidence_text = "\n".join(evidence_lines)

        confidence = self._compute_confidence(responses)
        has_conflict = self._detect_conflict(responses)

        queried_wids = sorted(set(wid for wid, _ in responses))
        not_queried = [i for i in range(len(chunks)) if i not in queried_wids]

        user_prompt = (
            f"Question: {question}\n\n"
            f"There are {len(chunks)} document chunks (Worker 0 ~ {len(chunks)-1}).\n"
            f"Current evidence:\n{evidence_text}\n\n"
            f"Current confidence: {confidence:.2f}\n"
            f"Conflict: {'yes' if has_conflict else 'no'}\n"
            f"Already queried Workers: {queried_wids if queried_wids else 'none'}\n"
            f"Not yet queried Workers: {not_queried}\n"
            f"Remaining Worker-call budget: {budget_left}\n"
            f"Round: {round_num}/{max_rounds}\n\n"
            "Available actions:\n"
            "  QUERY[i]  : ask Worker i (e.g., QUERY[3])\n"
            "  QUERY_ALL : broadcast to all Workers\n"
            "  VERIFY    : let conflicting Workers re-read swapped chunks and add an arbitrator\n"
            "  ANSWER    : synthesize the final answer from current evidence and stop\n"
            "  STOP      : stop scheduling\n\n"
            "Rules:\n"
            "- If at least one Worker already returned a concrete answer and there is NO conflict, output <action>ANSWER</action>.\n"
            "- Do NOT query a Worker that has already been queried.\n"
            "- Keep your thought short (one sentence). The action tag must appear at the end.\n\n"
            "Examples:\n"
            "<thought>Worker 3 returned a concrete answer and no other Worker contradicts it.</thought>\n"
            "<action>ANSWER</action>\n\n"
            "<thought>The queried Workers disagree, so I should verify before answering.</thought>\n"
            "<action>VERIFY</action>\n\n"
            "Now decide:\n"
            "<thought>...</thought>\n"
            "<action>...</action>"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_agent_action(self, text: str) -> tuple[str, int | None]:
        """从模型输出中解析动作。"""
        text = text.strip()
        match = re.search(r"<action>\s*([A-Z_]+)(?:\[(\d+)\])?\s*</action>", text)
        if not match:
            match = re.search(r"\b(QUERY|QUERY_ALL|VERIFY|ANSWER|STOP)\b(?:\[(\d+)\])?", text)
            if not match:
                return "ANSWER", None
        action = match.group(1)
        idx = int(match.group(2)) if match.group(2) is not None else None
        return action, idx

    def _execute_agent_action(
        self,
        action: str,
        idx: int | None,
        question: str,
        chunks: list[str],
        workers: list,
        responses: list[tuple[int, str]],
    ) -> tuple[list[tuple[int, str]], int, str]:
        """执行 Agent 动作，返回更新后的 responses、消耗的调用数、动作描述。"""
        calls = 0
        description = action

        if action == "QUERY" and idx is not None and 0 <= idx < len(workers):
            resp = workers[idx].answer(question, chunks[idx])
            responses.append((idx, resp))
            calls = 1
            description = f"QUERY[{idx}]"
        elif action == "QUERY_ALL":
            for chunk, worker in zip(chunks, workers):
                resp = worker.answer(question, chunk)
                responses.append((worker.worker_id, resp))
            calls = len(workers)
            description = "QUERY_ALL"
        elif action == "VERIFY":
            if self._detect_conflict(responses):
                responses = self._resolve_conflict(question, responses, chunks, workers)
                calls = 3
                description = "VERIFY"
            else:
                description = "VERIFY(no_conflict)"
        elif action in ("ANSWER", "STOP"):
            description = action
        else:
            description = "ANSWER(fallback)"

        return responses, calls, description

    def run_agent_loop(
        self,
        question: str,
        chunks: list[str],
        workers: list,
        seed_top_k: int = 5,
        max_rounds: int = 4,
        budget: int = 10,
    ) -> dict:
        """Phase 3：模型自己决定调度动作的多轮 loop。

        流程：
        1. 先用 BM25 召回 seed_top_k，给 Agent 初始证据；
        2. Agent 每轮根据当前状态选择一个动作；
        3. 执行动作，更新状态；
        4. 遇到 ANSWER/STOP 或预算耗尽时停止。
        """
        assert len(chunks) == len(workers), "chunks 与 workers 数量必须一致"

        responses: list[tuple[int, str]] = []
        round_history = []
        worker_calls = 0
        budget_left = budget

        seed_k = min(seed_top_k, len(chunks), budget_left)
        if seed_k > 0:
            seed_responses = self._selective_query(question, chunks, workers, seed_k)
            responses.extend(seed_responses)
            worker_calls += seed_k
            budget_left -= seed_k
            round_history.append({
                "round": 0,
                "type": "seed_selective_query",
                "question": question,
                "responses": list(seed_responses),
            })

        final_answer = None
        stop_reason = ""

        for round_num in range(1, max_rounds + 1):
            if budget_left <= 0:
                stop_reason = "budget_exhausted"
                break

            messages = self._build_agent_prompt(
                question, chunks, responses, budget_left, round_num, max_rounds
            )
            raw_output = apply_chat_and_generate(
                self.model,
                self.tokenizer,
                messages,
                max_new_tokens=128,
                do_sample=False,
            ).strip()

            action, idx = self._parse_agent_action(raw_output)
            responses, calls, description = self._execute_agent_action(
                action, idx, question, chunks, workers, responses
            )
            worker_calls += calls
            budget_left -= calls

            round_history.append({
                "round": round_num,
                "type": "agent_action",
                "raw_output": raw_output,
                "action": description,
                "responses": list(responses),
                "budget_left": budget_left,
            })

            if action in ("ANSWER", "STOP") or description.startswith("ANSWER"):
                stop_reason = "agent_ANSWER" if description.startswith("ANSWER") else f"agent_{action}"
                break

        if final_answer is None:
            final_answer = self.answer(question, responses)
            if not stop_reason:
                stop_reason = "max_rounds"

        return {
            "final_answer": final_answer,
            "worker_responses": responses,
            "worker_calls": worker_calls,
            "confidence": self._compute_confidence(responses),
            "rounds": round_history,
            "stop_reason": stop_reason,
        }
