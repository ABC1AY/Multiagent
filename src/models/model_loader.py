"""模型加载与推理工具：加载 Qwen2.5-3B-Instruct 并提供 chat + generate 接口。"""
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, MODEL_PATH

logger = logging.getLogger(__name__)

# 模块级缓存，避免重复加载到显存
_model_cache: Optional[tuple[PreTrainedModel, PreTrainedTokenizer]] = None


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载因果语言模型及其 tokenizer。

    返回值会被模块级缓存，后续调用直接返回同一实例，避免重复占用显存。

    Args:
        model_path: 本地模型目录。默认为 ``src.config.MODEL_PATH``。
        device_map: ``transformers`` 的 device_map 参数。
        dtype: 模型权重的数据类型。
        trust_remote_code: 是否信任模型仓库中的自定义代码。

    Returns:
        ``(model, tokenizer)`` 元组。
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    path = str(model_path) if model_path is not None else str(MODEL_PATH)
    logger.info(f"Loading model from: {path}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
    )
    # 确保 pad_token 可用（生成批量/单条输入时都可能用到）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    _model_cache = (model, tokenizer)
    return _model_cache


def apply_chat_and_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
) -> str:
    """把 chat messages 经 chat template 编码后生成回复。

    仅返回新生成的文本部分，不包含 prompt。

    Args:
        model: 因果语言模型。
        tokenizer: 对应的 tokenizer。
        messages: 符合 chat template 的消息列表，例如
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``。
        max_new_tokens: 最多生成的新 token 数。
        do_sample: 是否使用采样。若为 False，则使用贪心解码。
        temperature: 采样温度，仅在 ``do_sample=True`` 时生效。
        top_p: nucleus 采样的概率阈值，仅在 ``do_sample=True`` 时生效。
        add_generation_prompt: 是否在模板末尾追加生成提示符。

    Returns:
        模型生成的文本字符串（已去除 special tokens）。
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


def generate_with_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
) -> tuple[str, list[int], list[float]]:
    """生成文本并返回 (解码文本, token_ids, 逐token对数概率)。

    与 ``apply_chat_and_generate()`` 不同，本函数：

    1. **不** 使用 ``torch.no_grad()``，允许梯度回传用于训练。
    2. 通过 ``output_scores=True`` 获取每个生成 token 的 logits。
    3. 使用 ``torch.log_softmax`` 计算对数概率。
    4. 除了解码文本外，还返回 token ID 及其对数概率列表。

    Args:
        model: 因果语言模型。
        tokenizer: 对应的 tokenizer。
        messages: 符合 chat template 的消息列表。
        max_new_tokens: 最多生成的新 token 数。
        do_sample: 是否使用采样。若为 False，则使用贪心解码。
        temperature: 采样温度，仅在 ``do_sample=True`` 时生效。
        top_p: nucleus 采样的概率阈值，仅在 ``do_sample=True`` 时生效。
        add_generation_prompt: 是否在模板末尾追加生成提示符。

    Returns:
        ``(decoded_text, token_ids, log_probs)`` 元组，其中 ``token_ids``
        和 ``log_probs`` 均为列表，长度等于生成的 token 数。
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **generate_kwargs)

    # 提取生成的 token（排除 prompt 部分）
    sequences = outputs.sequences
    new_tokens = sequences[0][input_length:]

    # scores 是一个 tuple，每个元素对应一个生成步骤的 logits
    scores = outputs.scores

    # 计算实际生成的每个 token 的对数概率
    log_probs: list[float] = []
    for i, token_id in enumerate(new_tokens):
        logits = scores[i]  # shape: (1, vocab_size) or (vocab_size,)
        # 如果有 batch 维度，取第一个
        if logits.dim() == 2:
            logits = logits[0]
        log_probs_tensor = torch.log_softmax(logits, dim=-1)
        log_prob = log_probs_tensor[token_id].item()
        log_probs.append(log_prob)

    token_ids = new_tokens.tolist()
    decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decoded_text, token_ids, log_probs
