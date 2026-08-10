"""TriviaQA RC (Wikipedia context) loader — validation split only.

Produces a JSONL file with the fields used downstream:
    id, document, question, answer, aliases
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import DATA_PROCESSED_DIR, TRIVIAQA_PROCESSED_PATH


def _build_document(entity_pages: dict) -> str:
    """Concatenate all Wikipedia context pages into a single document string.

    TriviaQA RC's `entity_pages` is a dict of parallel lists:
        wiki_context: list[str]  — full Wikipedia section text
        title:        list[str]  — page titles
    We join them with section headers so the result reads like a
    multi-section Wikipedia article (similar in shape to the current
    multi-paragraph synthetic documents).
    """
    contexts = entity_pages.get("wiki_context", []) or []
    titles = entity_pages.get("title", []) or []
    if not contexts:
        return ""
    parts = []
    for title, ctx in zip(titles, contexts):
        if title:
            parts.append(f"== {title} ==\n{ctx.strip()}")
        else:
            parts.append(ctx.strip())
    return "\n\n".join(parts)


def load_triviaqa_rc(split: str = "validation"):
    """Load TriviaQA RC from HuggingFace using the token in .env.

    ``override=True`` ensures the project ``.env`` wins over any HF_*
    variables already present in the shell environment, so the China
    mirror (``HF_ENDPOINT``) and token configured in ``.env`` are always
    the ones actually used.
    """
    load_dotenv(project_root / ".env", override=True)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Add it to the project .env file: "
            "HF_TOKEN=hf_..."
        )
    ds = load_dataset(
        "mandarjoshi/trivia_qa",
        "rc",
        split=split,
        token=token,
    )
    return ds


def convert_to_jsonl(dataset, output_path: Path = TRIVIAQA_PROCESSED_PATH) -> list[dict]:
    """Write the HuggingFace dataset to our JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    skipped = 0
    for row in tqdm(dataset, desc="Converting TriviaQA → JSONL"):
        document = _build_document(row["entity_pages"])
        if not document:
            skipped += 1
            continue
        answer_obj = row["answer"]
        samples.append({
            "id": row["question_id"],
            "document": document,
            "question": row["question"],
            "answer": answer_obj["value"],
            "aliases": list(answer_obj.get("aliases", [])),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(samples)} samples -> {output_path} "
        f"(skipped {skipped} rows with empty wiki context)"
    )
    return samples


def ensure_triviaqa_dataset(
    output_path: Path = TRIVIAQA_PROCESSED_PATH,
    force_regenerate: bool = False,
    use_sample: bool = False,
) -> list[dict]:
    """Load from disk if present, otherwise download + convert.

    This mirrors the `ensure_dataset()` helper in the baseline scripts
    so swapping it in is a one-line change.

    If use_sample=True, loads the 5-sample test dataset instead (useful when
    network access to HuggingFace is unavailable).
    """
    from src.config import TRIVIAQA_SAMPLE_PATH

    if use_sample:
        sample_path = TRIVIAQA_SAMPLE_PATH
        if not sample_path.exists():
            print(f"Sample dataset not found. Generating {sample_path}...")
            from src.data_generation.sample_triviaqa import generate_sample_dataset
            generate_sample_dataset(sample_path)
        samples = []
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Loaded {len(samples)} sample TriviaQA samples from {sample_path}")
        return samples

    if output_path.exists() and not force_regenerate:
        samples = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Loaded {len(samples)} TriviaQA samples from {output_path}")
        return samples

    ds = load_triviaqa_rc(split="validation")
    return convert_to_jsonl(ds, output_path)


if __name__ == "__main__":
    ensure_triviaqa_dataset()
