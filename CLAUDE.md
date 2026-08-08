# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on GRPO-based budget-aware dynamic multi-agent scheduling for long document QA. The system implements a Leader-Worker architecture where a Leader agent coordinates multiple Worker agents to answer questions about long documents by selectively querying relevant document chunks.

**Current Status**: Phase 2 complete (hardcoded scheduling optimization), working toward Phase 3 (learnable coverage sufficiency estimation).

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Model Setup
```bash
# Download Qwen2.5-3B-Instruct model
python scripts/download_model.py

# Test model loading and inference
python scripts/test_model.py
```

### Running Experiments
```bash
# Phase 1: LONGAGENT-style baseline (broadcast to all workers)
python scripts/run_phase1_baseline.py

# Phase 2: Selective scheduling with BM25 retrieval
python scripts/run_phase2_baseline.py
```

### Data Generation
```bash
# Generate needle-in-haystack dataset (100 samples by default)
python src/data_generation/needle_in_haystack.py

# Setup TriviaQA dataset (real-world QA with Wikipedia context)
# 1. Add your HuggingFace token to .env
echo "HF_TOKEN=hf_..." >> .env

# 2. Install the new dependency
pip install python-dotenv

# 3. Download and convert the validation split (~8.8K QA pairs)
python src/data_generation/triviaqa_loader.py

# Alternative: Use sample dataset (5 samples, no network required)
python src/data_generation/sample_triviaqa.py
# Then set USE_SAMPLE = True in run_phase1_baseline.py and run_phase2_baseline.py
```

## Architecture

### Core Components

**Leader Agent** (`src/agents/leader.py`)
- Coordinates Worker agents and synthesizes final answers
- Implements two execution loops:
  - `run_multi_agent_loop()`: Phase 1 broadcast (queries all workers)
  - `run_selective_loop()`: Phase 2 selective querying with BM25 retrieval, optional conflict resolution, and fallback to broadcast
- Key methods: `_selective_query()`, `_resolve_conflict()`, `_compute_confidence()`

**Worker Agent** (`src/agents/worker.py`)
- Reads a single document chunk and answers questions
- Returns `NO_MENTION_MARKER` ("未提及") when information is not found
- All workers share the same model instance but maintain separate contexts

**BM25 Retriever** (`src/retrieval/bm25_retriever.py`)
- Token-based lightweight retrieval using BM25 algorithm
- Used by Leader to select top-K relevant chunks before querying workers
- Reduces worker calls from ~15 to ~5 (68% reduction) while maintaining accuracy

**Data Generation** (`src/data_generation/`)
- `needle_in_haystack.py`: Generates synthetic long documents with embedded facts
  - Target lengths: 4K / 8K / 16K / 32K tokens
  - Needle positions: front / middle / back
  - Fact templates: birthday, university, email, color
- `triviaqa_loader.py`: Loads TriviaQA RC dataset (Wikipedia context, validation split)
  - Downloads from HuggingFace using token in `.env`
  - Converts to JSONL format compatible with the pipeline
  - ~8.8K QA pairs with real Wikipedia articles as context
- `sample_triviaqa.py`: Generates 5-sample test dataset (no network required)
  - Use when HuggingFace is inaccessible
  - Set `USE_SAMPLE = True` in baseline scripts to use this dataset
- `chunking.py`: Token-based fixed-window document splitting with overlap

**Baselines** (`src/baselines.py`)
- `single_model_full_context()`: Full document in prompt (simulates long-context model)
- `single_model_truncated()`: First K tokens only (simulates fixed-length truncation)

**Evaluation** (`src/evaluation/metrics.py`)
- `contains_answer()`: Loose substring matching for short factual answers
- `compute_metrics()`: Calculates accuracy, token consumption, worker call counts

### Model Loading

The codebase references `src/models/model_loader.py` which should provide:
- `load_model_and_tokenizer()`: Loads Qwen2.5-3B-Instruct from `models/qwen2.5-3b-instruct/models/qwen--Qwen2.5-3B-Instruct/snapshots/master/`
- `apply_chat_and_generate()`: Applies chat template and generates responses

**Note**: This file is not yet implemented and needs to be created.

### Configuration

`src/config.py` defines:
- `PROJECT_ROOT`: Project root directory
- `MODEL_PATH`: Local model snapshot path
- `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`: Data directories
- `EXPERIMENTS_DIR`, `LOGS_DIR`, `RESULTS_DIR`, `CHECKPOINTS_DIR`: Experiment output directories
- Default generation parameters: `DEFAULT_MAX_NEW_TOKENS=128`, `DEFAULT_TEMPERATURE=0.7`
- Chunking defaults: `DEFAULT_CHUNK_SIZE=512`, `DEFAULT_CHUNK_OVERLAP=50`

## Experimental Roadmap

The project follows a 5-phase progression (do not skip phases):

- **Phase 0** ✅: Environment setup and model download
- **Phase 1** ✅: Simplified LONGAGENT baseline (broadcast to all workers)
- **Phase 2** ✅: Hardcoded scheduling optimization (BM25 retrieval + conflict resolution + fallback)
- **Phase 3** 🔄: Learnable coverage sufficiency estimation (train classifier to decide when to stop)
- **Phase 4**: GRPO end-to-end training of scheduling policy
- **Phase 5**: Public benchmarks (LongBench, InfiniteBench) and real documents

### Phase 2 Results (20 samples, Qwen2.5-3B-Instruct)

| Method | Accuracy | Avg Tokens | Avg Worker Calls | Time |
|--------|----------|------------|------------------|------|
| broadcast (Phase 1) | 75.0% | 7,905.5 | 15.1 | 102.9s |
| selective (BM25 top-5) | 75.0% | 2,486.1 | 4.7 | 38.4s |
| selective + conflict | 75.0% | 2,486.1 | 4.7 | 37.7s |
| selective + conflict + fallback | 75.0% | 3,350.4 | 6.3 | 47.9s |

BM25 retrieval alone reduces worker calls by 69% and token consumption by 68% while maintaining accuracy.

## Key Design Patterns

### Leader-Worker Communication
1. Leader builds prompts for workers with question + chunk
2. Workers return answers or "未提及" (not mentioned)
3. Leader synthesizes final answer from all worker responses
4. Confidence = max agreement ratio among non-"未提及" responses

### Selective Querying (Phase 2)
1. BM25 retriever ranks chunks by relevance to question
2. Leader queries only top-K workers (default K=5)
3. Optional conflict resolution: conflicting workers swap chunks and re-read
4. Optional fallback: if confidence < threshold, broadcast to all workers

### Token Accounting
Token counts are approximated by counting tokens in reconstructed prompts (question + chunk + system prompt). This is tracked in `_count_tokens_in_result()` for Phase 2 and inline for Phase 1.

## Important Notes

- **Model path**: Hardcoded in `src/config.py` and `scripts/test_model.py` - update if model location changes
- **Shared model instance**: Leader and all Workers use the same model loaded once to save GPU memory
- **CUDA OOM handling**: `baselines.py` catches `torch.OutOfMemoryError` and returns placeholder answers
- **Dataset format**: JSONL with fields depending on source:
  - `needle_in_haystack.jsonl`: `id`, `document`, `question`, `answer`, `needle`, `target_length`, `position` (synthetic Chinese QA)
  - `triviaqa_rc.jsonl`: `id`, `document`, `question`, `answer`, `aliases` (TriviaQA RC Wikipedia context, validation split)
  - Only `document`, `question`, `answer` are used by the pipeline; other fields are metadata
- **Results format**: JSON with metrics, predictions, timing, and configuration parameters
