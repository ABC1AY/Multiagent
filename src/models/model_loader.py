"""统一的本地模型加载接口。"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen2.5-3b-instruct"
    / "models"
    / "qwen--Qwen2.5-3B-Instruct"
    / "snapshots"
    / "master"
)


def load_model_and_tokenizer(
    model_path: Path | str | None = None,
    load_in_4bit: bool = False,
    dtype: torch.dtype = torch.float16,
):
    """加载模型与分词器。

    Args:
        model_path: 模型本地路径，默认使用 MODEL_PATH。
        load_in_4bit: 是否使用 4-bit 量化，8G 显存下多实例运行建议开启。
        dtype: 默认 float16。
    """
    model_path = Path(model_path) if model_path else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run scripts/download_model.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_compute_dtype"] = dtype
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer


def apply_chat_and_generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float | None = None,
    do_sample: bool = False,
) -> str:
    """使用 chat template 生成回复。"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if do_sample and temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return response.strip()

