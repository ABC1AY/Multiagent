# GRPO Training Readiness Audit — Foundational Implementation Checklist

- Record format: `2`
- Mode: `session-documentation`
- Date: `2026-08-10`
- Project: /home/mzhyui/git/MultiagentLongText
- Status: `concluded`
- Evidence state: `verified`

## Outcome

A systematic audit of the codebase identified **14 foundational items** across 4 tiers that must be addressed before GRPO end-to-end training of the scheduling policy can begin. The current pipeline is **inference-only**: generation runs under `torch.no_grad()` and returns decoded text without log probabilities (E4); scheduling decisions are hardcoded heuristics (E2, E6); no reward function exists beyond binary accuracy (E5); and `src/training/` is empty (E9). Four items are **hard blockers** (GRPO cannot function without them): log-probability extraction, sampling-based rollout, a scalar reward function, and episode trajectory recording. Three more are **high-impact prerequisites**: a learnable coverage-sufficiency estimator (Phase 3, the explicit roadmap predecessor), a formal policy action-space definition, and a reference-model + gradient-flow architecture. The remaining items cover baseline revalidation and training infrastructure.

## Context and Scope

Request: "According to `progress/2026-08-09-phase2-triviaqa-evaluation.md` and current multi-agent design, before we start the GRPO training to improve the agent accuracy, what foundational implementation should be checked and ensured?"

Scope: Read-only audit of the existing codebase against GRPO requirements. No code was written or modified. The analysis covers all Python source files in `src/`, the `scripts/` directory, `requirements.txt`, and `CLAUDE.md`.

Starting state:
- Phase 2 complete with TriviaQA evaluation on 20 samples (E1)
- Phase 1 and Phase 2 baselines run successfully; BM25 selective scheduling drops accuracy on TriviaQA (35% vs 45% broadcast) (E1)
- `_resolve_conflict` bug fixed (E2)
- Roadmap places Phase 3 (learnable coverage sufficiency) before Phase 4 (GRPO training) (E11)
- `src/training/` directory exists but contains only `__init__.py` (E9)

Exclusions: No implementation, no experiments run, no needle-in-haystack revalidation (only noted as needed).

## Findings and Decisions

### Tier 1 — Critical Blockers (GRPO cannot run without these)

**F1. Log-Probability Extraction** `verified` (E4)

`apply_chat_and_generate` in `src/models/model_loader.py` wraps generation in `torch.no_grad()` and returns only `tokenizer.decode(new_tokens, skip_special_tokens=True)`. GRPO requires token-level log probabilities `log p(a_t | s_t)` for every action in a rollout trajectory. A parallel function — e.g., `generate_with_logprobs()` returning `(text, token_ids, log_probs)` — must be implemented. The `model.generate()` API supports `output_scores=True` and `return_dict_in_generate=True` to recover per-token log-probs via `torch.log_softmax` on the logits.

**F2. Sampling-Based Generation for Rollouts** `verified` (E2, E3)

Both Worker and Leader currently use `do_sample=False` (greedy decoding) exclusively (E3, E4). GRPO rollouts require stochastic sampling to generate G diverse trajectories per prompt and compute group-relative advantages. A verified sampling path with configurable `temperature` and `top_p`, plus the ability to produce G independent rollouts per (question, document) pair, must be implemented and tested.

**F3. Scalar Reward Function** `verified` (E5)

The current evaluation provides `contains_answer()` (binary substring match) and `compute_metrics()` for reporting (E5). GRPO needs a per-rollout scalar reward. For a budget-aware scheduling system, the reward should combine:

- Accuracy component: binary correctness or partial match score
- Efficiency penalty: negative cost proportional to token consumption and/or worker calls
- A balance weight λ to control the accuracy–efficiency trade-off

A `compute_reward(prediction, reference, token_cost, worker_calls) → float` function must be defined.

**F4. Episode Trajectory Recorder** `inferred` (E2, E10)

GRPO trains on complete trajectories. The current `run_selective_loop` (E2) executes the pipeline and records round metadata, but not in a training-compatible format. An episode recorder must capture, for each decision point: the state (question, retrieved chunks, accumulated worker responses), the action taken (which chunks to query, whether to stop, whether to conflict-resolve), the log-probability of that action, and the final scalar reward attributed to the episode.

### Tier 2 — High-Impact Prerequisites (determine training quality)

**F5. Phase 3 Learnable Coverage-Sufficiency Estimator** `verified` (E1, E11)

The roadmap explicitly sequences Phase 3 before Phase 4 (E11). The Phase 2 results strongly motivate this: BM25 selective scheduling drops accuracy on TriviaQA from 45% to 35% (E1). With `confidence_threshold=1.0` and fallback, the system degenerates to full broadcast for most TriviaQA questions. A learned stopping criterion — predicting "coverage sufficient" vs "need more chunks" — provides the key learnable decision that GRPO can later optimize end-to-end.

**F6. Policy Action-Space Formalization** `inferred` (E2, E6, E11)

The current Leader makes scheduling decisions through hardcoded logic: BM25 top-K retrieval (E6), fixed confidence thresholds, and unconditional fallback (E2). For GRPO, the learnable actions must be explicitly defined. Two candidate approaches:

- **Option A — End-to-end generation**: The model generates scheduling commands as text tokens (e.g., "query chunks 3, 7, 12" or "stop"). Flexible but unconstrained and harder to parse reliably.
- **Option B — Structured heads**: Separate linear heads on top of model hidden states produce (1) chunk selection scores, (2) stop/go probability, (3) conflict resolution trigger. More constrained, easier to train, but less flexible.

This architectural decision shapes the entire GRPO training setup and should be resolved before implementation.

**F7. Reference Model Setup** `inferred` (E4)

GRPO requires a frozen reference model for KL divergence computation between the current policy and the reference distribution. The current codebase loads a single cached model instance (E4). A mechanism to maintain two copies is needed: an active policy (with gradients) and a frozen reference (no gradients, periodic sync). If LoRA is adopted, the reference becomes the base weights while the policy is base + adapter, reducing memory overhead.

**F8. Gradient-Flow Architecture** `verified` (E4)

The entire inference path is wrapped in `torch.no_grad()` (E4). For GRPO training, gradients must flow through at least the policy parameters. The multi-hop pipeline (Worker generation → Leader synthesis) creates a long computation graph. Three options:

- **Full fine-tuning**: all 3B parameters — expensive memory and compute
- **LoRA** (recommended for 3B model): train low-rank adapters only, base model frozen — efficient and standard for GRPO
- **Hybrid**: LoRA on model weights, learnable linear heads for scheduling decisions

### Tier 3 — Baseline Revalidation (ensure solid foundation)

**F9. Re-run Needle-in-Haystack After Bug Fix** `verified` (E1)

The `_resolve_conflict` bug fix was validated only on TriviaQA. The Phase 2 progress notes explicitly state: "needle-in-haystack and other benchmarks not re-run after bug fix" (E1). Phase 1 and Phase 2 should be re-run on needle-in-haystack to confirm no regression before the results are used as training data or reward calibration.

**F10. Larger-Scale Evaluation for Statistical Significance** `verified` (E1)

Phase 2 results are based on 20 samples (E1). The 10-percentage-point difference (45% vs 35%) may not be statistically significant. Running on 100+ samples is recommended before using results as training data labels or for reward function calibration.

**F11. Precise Token Accounting** `inferred` (E10)

The current `_count_tokens_in_result()` in `scripts/run_phase2_baseline.py` reconstructs approximate prompts from fragments for post-hoc estimation (E10). For GRPO reward computation, token counting should be exact and recorded inline during each generation step of the rollout, not estimated after the fact.

### Tier 4 — Training Infrastructure

**F12. Training Dependencies** `verified` (E8)

Current `requirements.txt` includes `torch`, `transformers`, `accelerate`, `bitsandbytes`, `datasets` — a solid inference foundation (E8). GRPO training typically additionally requires:
- `trl` (Hugging Face TRL library) or a custom GRPO implementation
- `peft` (for LoRA adapters)
- `wandb` or `tensorboard` (for training logs)

**F13. Training Module Structure** `verified` (E9)

`src/training/` exists with only an empty `__init__.py` (E9). The following modules are needed:
- `grpo_trainer.py` — main GRPO training loop
- `rollout.py` — episode generation (calls the multi-agent pipeline with log-prob tracking)
- `reward.py` — scalar reward computation
- `policy.py` — policy definition (action space, parameterization)
- `dataset.py` — training data pipeline

**F14. Checkpointing and Experiment Tracking** `inferred` (E7)

`CHECKPOINTS_DIR` is defined in `src/config.py` but never used (E7). Training needs periodic checkpoint saves (model + optimizer + scheduler state), validation evaluation at each checkpoint, training curves (loss, reward, accuracy over steps), and hyperparameter logging.

## Technical Design or Experimental Plan

### Proposed Implementation Priority (Critical Path)

```
Phase A — Inference Primitives (blockers F1–F4):
  1. generate_with_logprobs() in model_loader.py
  2. Verified sampling path (G rollouts per prompt)
  3. compute_reward() function in src/training/reward.py
  4. Episode trajectory recorder in src/training/rollout.py

Phase B — Policy Architecture (prerequisites F5–F8):
  5. Phase 3 learnable stopping estimator
  6. Policy action-space decision (Option A vs B)
  7. Reference model + LoRA setup
  8. Gradient-flow architecture

Phase C — Validation & Infrastructure (F9–F14):
  9.  Re-run baselines on needle-in-haystack
  10. Scale evaluation to 100+ samples
  11. Precise per-step token accounting
  12–14. Training deps, module structure, checkpointing
```

### Key Interfaces to Implement

**Log-prob generation** (proposed signature):
```python
def generate_with_logprobs(
    model, tokenizer, messages,
    max_new_tokens=128, do_sample=True,
    temperature=0.7, top_p=0.9,
) -> tuple[str, list[int], list[float]]:
    """Returns (decoded_text, token_ids, per_token_log_probs)."""
```

**Reward function** (proposed signature):
```python
def compute_reward(
    prediction: str,
    reference: str,
    token_cost: int,
    worker_calls: int,
    accuracy_weight: float = 1.0,
    efficiency_weight: float = 0.1,
) -> float:
    """Scalar reward: +accuracy_weight if correct, -efficiency_weight * cost."""
```

### Reward Design Considerations

The reward should incentivize the policy to find the correct answer using minimal resources. A simple formulation:

```
R = accuracy_weight * contains_answer(pred, ref) - efficiency_weight * (token_cost / max_budget)
```

Where `max_budget` is a normalization constant (e.g., the token cost of full broadcast). This produces rewards in approximately [−1, +1]:
- Perfect answer with zero cost: +1.0
- Perfect answer at full budget: +1.0 − efficiency_weight
- Wrong answer with zero cost: 0.0
- Wrong answer at full budget: −efficiency_weight

## Evidence Ledger

| ID | Class | Locator / Check | Conclusion |
|----|-------|----------------|------------|
| E1 | verified | `progress/2026-08-09-phase2-triviaqa-evaluation.md` (SHA-256: `2f90032f5548faaf14e636272fea39e92d50ca598412bf6de5b07032f11aea0f`) | Phase 2 TriviaQA results: selective 35% vs broadcast 45%; bug fix applied; 20-sample evaluation |
| E2 | verified | `src/agents/leader.py` (SHA-256: `3be8fcfcd5956c72541589b87692223027dae5b547a3f5d1527a9f98ed7c75a2`) | Leader scheduling decisions are hardcoded heuristics; no trainable policy interface |
| E3 | verified | `src/agents/worker.py` (SHA-256: `d550b60c5b1d448f38a77706abaa65b77aa860e930463ac9f2ffc13968059110`) | Worker uses greedy decoding (do_sample=False); no log-prob or sampling support |
| E4 | verified | `src/models/model_loader.py` (SHA-256: `e388dfb210b5bdf3ee5f6cce5eaedf7955e273dfbe952a13f36f90b8914e0a29`) | apply_chat_and_generate uses torch.no_grad(), returns only decoded text; no log-prob extraction |
| E5 | verified | `src/evaluation/metrics.py` (SHA-256: `a7af479dbe7360da5af5839f78db23934ac9bd85e5542a5c31ebec813de370b7`) | Evaluation provides binary accuracy only; no scalar reward function for training |
| E6 | verified | `src/retrieval/bm25_retriever.py` (SHA-256: `3b2cdf9d7d48ad7c390e0a5a7b99dfee3f92a0b39bf220d581dc9db26beed6e0`) | BM25 chunk selection is hardcoded top-K; not learnable |
| E7 | verified | `src/config.py` (SHA-256: `1f9e2e024e4f570f2a26a1725f08ca462e2575f0820c80295a4dbf7a728b43ac`) | CHECKPOINTS_DIR defined but unused; no training configuration |
| E8 | verified | `requirements.txt` (SHA-256: `2a7e8564a678691ad0309d9cd83d33045b337a3b3356d20c99088394ba4de69c`) | No trl, peft, or vllm dependencies for GRPO training |
| E9 | verified | `src/training/__init__.py` (empty, no hash) | Training module directory exists but contains no implementation |
| E10 | verified | `scripts/run_phase2_baseline.py` (SHA-256: `4ab8f07afd6b4b9baa735834b18258c29ba51cdfbdd9271af95eeccf8d207e52`) | Token accounting is post-hoc approximation, not precise per-step tracking |
| E11 | verified | `CLAUDE.md` (SHA-256: `e383dfcf783209c4f0373e0e4643d1ccc94baf7bbc94e7e7c699f9286676118b`) | Roadmap: Phase 3 (learnable coverage) before Phase 4 (GRPO training) |
| E12 | verified | `src/data_generation/chunking.py` (SHA-256: `0908bd05df7fcc5da68b303a588baba4579bfc2c03eca7e0aa430da16cd7d3a2`) | Token-based fixed-window chunking with overlap; stable utility |
| E13 | verified | `src/baselines.py` (SHA-256: `0983da21230ca18dcd17f2ac99c8f571477c4539d728274087508abfd6b7f921`) | Single-model baselines with OOM handling; stable reference points |

## Evidence Boundary

**Established:**
- The current inference pipeline lacks all four GRPO primitives (log-probs, sampling, reward, trajectories)
- Scheduling decisions are entirely hardcoded and not parameterized for learning
- Phase 3 (learnable coverage sufficiency) is the explicit roadmap prerequisite for Phase 4
- The training module is empty; no training infrastructure exists
- Dependencies for GRPO (trl, peft) are not installed
- 14 specific items identified across 4 priority tiers

**Not established:**
- Whether the policy should use end-to-end generation or structured heads (Option A vs B in F6)
- Whether LoRA or full fine-tuning is more appropriate given GPU constraints
- Exact reward function formulation and hyperparameters
- Whether Hugging Face TRL's `GRPOTrainer` can be adapted to the multi-agent pipeline or if custom implementation is needed
- Training time and resource requirements for 3B model with GRPO
- Generalizability of findings beyond TriviaQA and needle-in-haystack

## Next Steps

1. **Implement `generate_with_logprobs()`** in `src/models/model_loader.py` — the single highest-priority item; all downstream training code depends on it
2. **Design and implement `compute_reward()`** in `src/training/reward.py` — define the accuracy–efficiency trade-off
3. **Build episode trajectory recorder** in `src/training/rollout.py` — instrument the Leader pipeline to record (state, action, log-prob, reward) tuples
4. **Resolve policy action-space design** (Option A vs Option B) — architectural decision that shapes all training code
5. **Implement Phase 3 learnable coverage-sufficiency estimator** — roadmap prerequisite for GRPO
6. **Re-run Phase 1 and Phase 2 baselines** on needle-in-haystack to confirm no regression after bug fix
7. **Add training dependencies** (`trl`, `peft`, `wandb`) to `requirements.txt`
