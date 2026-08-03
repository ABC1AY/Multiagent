"""文档分块工具。"""
from transformers import PreTrainedTokenizer


def chunk_document(
    document: str,
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """按 token 数量对文档进行固定窗口切分。

    Args:
        document: 原始文档。
        tokenizer: 分词器。
        chunk_size: 每个 chunk 的最大 token 数。
        overlap: 相邻 chunk 之间的重叠 token 数。

    Returns:
        chunk 文本列表。
    """
    tokens = tokenizer.encode(document, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """统计文本的 token 数。"""
    return len(tokenizer.encode(text, add_special_tokens=False))

