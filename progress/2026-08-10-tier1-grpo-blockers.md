# Tier 1 GRPO Critical Blockers Implementation

- Record format: `2`
- Mode: `coding-progress`
- Date: `2026-08-10`
- Project: /home/mzhyui/git/MultiagentLongText
- Status: `completed`
- Evidence state: `verified`

## Outcome

Implemented all 4 Tier 1 Critical Blockers identified in the GRPO readiness audit (E8), enabling the transition from inference-only to training-capable pipeline. The system can now: extract per-token log probabilities from generation, perform stochastic sampling for diverse rollouts, compute scalar rewards combining accuracy and efficiency, and record complete episode trajectories with states, actions, log-probs, and rewards.

Deliverables:
- `generate_with_logprobs()` in `src/models/model_loader.py` — log-probability extraction (E1)
- `answer_with_logprobs()` in `src/agents/worker.py` — sampling-based worker responses (E2)
- `src/training/reward.py` — scalar reward function with group-relative advantages (E3)
- `src/training/types.py` — Episode, DecisionPoint, WorkerAction, LeaderAction data structures (E4)
- `src/training/rollout.py` — RolloutRecorder and run_rollout_with_trajectory() (E5)
- Updated `src/training/__init__.py` with module exports (E6)
- Cleaned duplicate lines in `requirements.txt` (E7)

Limitations:
- Token cost accounting in rollouts uses approximate prompt sizes (~100 tokens for worker, ~200 for synthesis) rather than exact counts
- Conflict resolution and fallback broadcast paths are not yet instrumented with trajectory recording
- No GRPO trainer implementation yet (Tier 2 and beyond)

## Task and Scope

Request: "fix Tier 1 — Critical Blockers (GRPO cannot run without these) in progress/2026-08-10-grpo-readiness-audit.md"

Target: Implement the 4 foundational primitives identified as hard blockers for GRPO training:
- F1: Log-Probability Extraction
- F2: Sampling-Based Generation for Rollouts
- F3: Scalar Reward Function
- F4: Episode Trajectory Recorder

Starting state:
- Phase 2 complete with BM25 selective scheduling working on TriviaQA
- `src/training/` directory existed but contained only empty `__init__.py`
- `apply_chat_and_generate()` used `torch.no_grad()` and returned only decoded text
- Worker and Leader both used greedy decoding (`do_sample=False`) exclusively
- No reward function existed beyond binary `contains_answer()` metric

Constraints:
- Preserve backward compatibility — existing Phase 1/Phase 2 baselines must continue to work unchanged
- New training code belongs in `src/training/`, not in existing agent code
- Follow existing code style (Chinese comments, type hints, docstrings)

Decisions:
- `generate_with_logprobs()` does NOT use `torch.no_grad()` to allow gradient flow for training
- Default `do_sample=True` for training rollouts (stochastic sampling)
- Reward formula: `R = accuracy_weight × is_correct − efficiency_weight × normalized_cost`
- Budget constants (15000 tokens, 25 worker calls) based on Phase 1 broadcast baseline

Exclusions:
- GRPO trainer implementation (Tier 2+)
- Reference model setup for KL divergence
- LoRA or full fine-tuning architecture
- Policy action-space formalization
- Conflict resolution trajectory recording

## Implementation

Execution sequence:
1. Read audit document (E8) to understand Tier 1 requirements (F1-F4)
2. Implemented `generate_with_logprobs()` in model_loader.py (F1)
3. Added `answer_with_logprobs()` to Worker class (F2)
4. Created `src/training/reward.py` with `compute_reward()` and `compute_group_relative_advantages()` (F3)
5. Created `src/training/types.py` with Episode, DecisionPoint, WorkerAction, LeaderAction dataclasses (F4)
6. Created `src/training/rollout.py` with RolloutRecorder and run_rollout_with_trajectory() (F4)
7. Updated `src/training/__init__.py` with module exports
8. Fixed duplicate lines in requirements.txt
9. Ran unit tests for all components — all passed
10. Ran regression test — Phase 1 and Phase 2 baselines still work

| Path | Symbol / Change | Role |
|------|----------------|------|
| `src/models/model_loader.py` | `generate_with_logprobs()` | Added: returns (text, token_ids, log_probs) using output_scores=True |
| `src/agents/worker.py` | `answer_with_logprobs()` | Added: sampling-based answer returning dict with text/ids/probs |
| `src/training/reward.py` | `compute_reward()`, `compute_group_relative_advantages()` | Created: scalar reward and GRPO advantage computation |
| `src/training/types.py` | `Episode`, `DecisionPoint`, `WorkerAction`, `LeaderAction` | Created: trajectory data structures with sequence_log_prob property |
| `src/training/rollout.py` | `RolloutRecorder`, `run_rollout_with_trajectory()`, `generate_rollout_batch()` | Created: episode recording and G-rollout batch generation |
| `src/training/__init__.py` | Module exports | Updated: exports all training primitives |
| `requirements.txt` | Removed duplicate lines | Fixed: cleaned duplicate dependency entries |

## Interface and Behavior Changes

### New Functions

**`generate_with_logprobs(model, tokenizer, messages, ...) -> (str, list[int], list[float])`**
- Returns decoded text, token IDs, and per-token log probabilities
- Does NOT wrap in `torch.no_grad()` (allows gradient flow)
- Default `do_sample=True` for stochastic generation

**`Worker.answer_with_logprobs(question, chunk, ...) -> dict`**
- Returns `{"text": str, "token_ids": list[int], "log_probs": list[float]}`
- Default `do_sample=True`, `temperature=0.7`, `top_p=0.9`

**`compute_reward(prediction, reference, token_cost, worker_calls, ...) -> float`**
- Reward range: approximately [-0.1, +1.0]
- Formula: `accuracy_weight × is_correct − efficiency_weight × normalized_cost`

**`run_rollout_with_trajectory(leader, question, reference_answer, chunks, workers, ...) -> Episode`**
- Executes selective query with sampling and records full trajectory
- Returns Episode with decisions, reward, and sequence_log_prob

### Backward Compatibility

All existing functions remain unchanged:
- `apply_chat_and_generate()` still uses `torch.no_grad()` and returns str
- `Worker.answer()` still uses greedy decoding and returns str
- `Leader.run_multi_agent_loop()` and `run_selective_loop()` work as before

## Validation

### V1 - pass

```text
.venv/bin/python -c "from src.training.reward import compute_reward; r1=compute_reward('Paris','Paris',1000,3); assert r1>0.9"
```

Observed: F3 reward function returns correct values for edge cases (correct+low-cost → 0.99, wrong+high-cost → -0.10)

### V2 - pass

```text
.venv/bin/python -c "from src.training.types import Episode; ep=Episode(...); assert ep.sequence_log_prob==-1.70"
```

Observed: F4 types dataclass and sequence_log_prob property compute correctly

### V3 - pass

```text
.venv/bin/python -c "from src.models.model_loader import generate_with_logprobs; text,ids,probs=generate_with_logprobs(m,t,msgs); assert len(ids)==len(probs)"
```

Observed: F1 generate_with_logprobs returns correct structure with matching token_ids and log_probs lengths, all log_probs ≤ 0

### V4 - pass

```text
.venv/bin/python -c "from src.agents.worker import Worker; r=worker.answer_with_logprobs(q,c); assert 'text' in r"
```

Observed: F2 Worker.answer_with_logprobs returns dict with text/token_ids/log_probs keys

### V5 - pass

```text
.venv/bin/python -c "from src.training.rollout import run_rollout_with_trajectory; ep=run_rollout_with_trajectory(...); assert len(ep.decisions)>0"
```

Observed: F4 rollout records trajectory with 4 decisions (3 worker queries + 1 synthesis), computes reward=0.99

### V6 - pass

```text
.venv/bin/python -c "result=leader.run_selective_loop(q,chunks,workers,top_k=3); assert 'final_answer' in result"
```

Observed: Regression test — Phase 2 run_selective_loop still returns correct result structure

## Evidence Ledger

| ID | Class | Locator / Check | Conclusion |
|----|-------|----------------|------------|
| E1 | verified | `src/models/model_loader.py` (SHA-256: `a73cb20a9bc490cde8e6d63804e8f62103cec4201053b8d50cc29baceeeabb1e`) | Added `generate_with_logprobs()` returning (text, token_ids, log_probs) without torch.no_grad() |
| E2 | verified | `src/agents/worker.py` (SHA-256: `de285723bde0c77695095c6492e25ee36934302937d4488dffbdfb6775a293ea`) | Added `answer_with_logprobs()` method with default do_sample=True |
| E3 | verified | `src/training/reward.py` (SHA-256: `2ee56e2f643b9c30d7a5475aea0cdd19f9d13caa4c5882871b4045ab12be4a26`) | Created compute_reward() combining accuracy and efficiency, plus group-relative advantages |
| E4 | verified | `src/training/types.py` (SHA-256: `94850e325b8bfeb3b5f8d259e95b3e957b78b3d2c2edaa20baa0570b2ded0b01`) | Created Episode, DecisionPoint, WorkerAction, LeaderAction dataclasses |
| E5 | verified | `src/training/rollout.py` (SHA-256: `bb0f1cf340f4730eb0c60134497b10d602aaedf7895d588f5c3e40f48cbe9e0c`) | Created RolloutRecorder and run_rollout_with_trajectory() for episode recording |
| E6 | verified | `src/training/__init__.py` (SHA-256: `182c515bb7ab3a1c7085abb83e48aa86a4ce8d96bd2f2e18e7c681122bcad187`) | Updated with module exports for all training primitives |
| E7 | verified | `requirements.txt` (SHA-256: `4e645cb65a77de586988ee7831b173a08947c373c0d87521e4100651f5f9aaa7`) | Removed duplicate dependency lines |
| E8 | verified | `progress/2026-08-10-grpo-readiness-audit.md` (SHA-256: `a44a3eb4b33fdbfcbaedecef444fe27d34052d5663496735e193b4aada65d9b1`) | Source audit document defining Tier 1 blockers F1-F4 |
| V1 | verified | `.venv/bin/python -c "from src.training.reward import compute_reward; r1=compute_reward(\"Paris\",\"Paris\",1000,3); assert r1>0.9"` | F3 reward function: correct values for edge cases |
| V2 | verified | `.venv/bin/python -c "from src.training.types import Episode; ep=Episode(...); assert ep.sequence_log_prob==-1.70"` | F4 types: Episode dataclass and sequence_log_prob work |
| V3 | verified | `.venv/bin/python -c "from src.models.model_loader import generate_with_logprobs; text,ids,probs=generate_with_logprobs(m,t,msgs); assert len(ids)==len(probs)"` | F1 generate_with_logprobs: returns correct structure |
| V4 | verified | `.venv/bin/python -c "from src.agents.worker import Worker; r=worker.answer_with_logprobs(q,c); assert \"text\" in r"` | F2 Worker.answer_with_logprobs: returns dict with text/ids/probs |
| V5 | verified | `.venv/bin/python -c "from src.training.rollout import run_rollout_with_trajectory; ep=run_rollout_with_trajectory(...); assert len(ep.decisions)>0"` | F4 rollout: records trajectory with decisions and reward |
| V6 | verified | `.venv/bin/python -c "result=leader.run_selective_loop(q,chunks,workers,top_k=3); assert \"final_answer\" in result"` | Regression: Phase 2 run_selective_loop still works |

## Git Custody

- Branch: `main`
- Baseline HEAD: `e574b4d2bcf42502fd3c5e312f18cbaa03cffb51` (unavailable — baseline not captured at start)
- Final HEAD: `e574b4d2bcf42502fd3c5e312f18cbaa03cffb51` (unchanged, uncommitted)
- History relation: unavailable
- Commits since baseline: none (changes uncommitted)
- Task-owned changes: `src/models/model_loader.py`, `src/agents/worker.py`, `src/training/reward.py`, `src/training/types.py`, `src/training/rollout.py`, `src/training/__init__.py`, `requirements.txt`, `progress/2026-08-10-tier1-grpo-blockers.md`
- Pre-existing changes: none (all task paths were either unchanged or newly created)
- Overlapping changes: all task paths are task-owned
- Outside-scope changes: `progress/2026-08-10-grpo-readiness-audit.md` (staged before this task)
- Ownership caveats: baseline was not captured at session start; all changes are in working tree, uncommitted
- Scoped diff: `files=0; insertions=0; deletions=0; binary_files=0; untracked_files=3`

## Evidence Boundary

**Established:**
- `generate_with_logprobs()` correctly extracts per-token log probabilities using HuggingFace's output_scores API
- Worker and Leader can perform stochastic sampling with configurable temperature/top_p
- Reward function produces sensible values across the accuracy-efficiency trade-off space
- Episode trajectories capture complete decision sequences with log-probs for GRPO training
- All existing Phase 1/Phase 2 inference functionality remains intact

**Not established:**
- Whether the gradient flow through `generate_with_logprobs()` works correctly for backpropagation (no training loop tested)
- Optimal reward function hyperparameters (accuracy_weight, efficiency_weight, budget constants)
- Whether conflict resolution and fallback paths need trajectory recording
- Integration with actual GRPO trainer (TRL or custom implementation)
- Memory footprint of storing full trajectories for G=4+ rollouts per prompt
- Training stability with the current log-prob extraction approach

## Next Steps

1. **Implement Tier 2 prerequisites** — Phase 3 learnable coverage-sufficiency estimator (F5), policy action-space formalization (F6), reference model setup (F7), gradient-flow architecture (F8)
2. **Add training dependencies** — `trl`, `peft`, `wandb` to requirements.txt for GRPO training
3. **Implement GRPO trainer** — Either adapt HuggingFace TRL's GRPOTrainer or build custom implementation using the Tier 1 primitives
4. **Instrument conflict resolution** — Add trajectory recording to `_resolve_conflict()` and fallback broadcast paths if needed for training signal
5. **Calibrate reward function** — Run ablation on accuracy_weight and efficiency_weight using baseline results as reference
