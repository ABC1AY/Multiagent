"""基于 token 的轻量 BM25 检索器，用于召回相关 chunk。"""
import math
from collections import Counter

from transformers import PreTrainedTokenizer


class BM25Retriever:
    """简单 BM25 实现，直接用分词器将文本转为 token id 进行检索。"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self._corpus_tokens: list[list[int]] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def fit(self, documents: list[str]) -> "BM25Retriever":
        """对文档集合构建 BM25 统计量。"""
        self._corpus_tokens = [
            self._tokenize(doc) for doc in documents
        ]
        self._doc_lengths = [len(tokens) for tokens in self._corpus_tokens]
        self._avgdl = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        self._n_docs = len(documents)

        for tokens in self._corpus_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self._doc_freqs[t] += 1

        return self

    def _tokenize(self, text: str) -> list[int]:
        """使用分词器编码为 token id 列表。"""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _idf(self, token: int) -> float:
        """计算 IDF。"""
        n = self._doc_freqs.get(token, 0)
        return math.log(
            (self._n_docs - n + 0.5) / (n + 0.5) + 1.0
        )

    def score(self, query: str, doc_idx: int) -> float:
        """计算单个文档与查询的 BM25 分数。"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._corpus_tokens[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        freq = Counter(doc_tokens)

        score = 0.0
        for token in query_tokens:
            f = freq.get(token, 0)
            if f == 0:
                continue
            idf = self._idf(token)
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 top-k 相关文档的 (索引, 分数) 列表。"""
        if self._n_docs == 0:
            return []

        scores = [
            (idx, self.score(query, idx))
            for idx in range(self._n_docs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_document_tokens(self, doc_idx: int) -> list[int]:
        """获取某个文档的 token 列表。"""
        return self._corpus_tokens[doc_idx]

