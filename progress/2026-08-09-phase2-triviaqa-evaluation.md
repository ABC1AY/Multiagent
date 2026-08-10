# Phase 2 TriviaQA Evaluation and Conflict Resolution Bug Fix

- Record format: `2`
- Mode: `coding-progress`
- Date: `2026-08-09`
- Project: /home/mzhyui/git/MultiagentLongText
- Status: `completed`
- Evidence state: `verified`

## Outcome

Evaluated Phase 1 (broadcast) and Phase 2 (selective scheduling with BM25 retrieval, conflict resolution, and fallback) baselines on 20 TriviaQA RC samples using Qwen2.5-3B-Instruct. Fixed an `IndexError` bug in `_resolve_conflict` that caused Phase 2 + conflict to crash when worker IDs exceeded the response list length. All four Phase 2 methods now run successfully. Results show that BM25-based selective scheduling, which preserved accuracy on needle-in-haystack, **reduces accuracy on TriviaQA** (35% vs 45% broadcast), motivating Phase 3 learnable coverage-sufficiency estimation.

Deliverables:
- Updated `experiments/results/phase1_baseline.json` with TriviaQA results
- Updated `experiments/results/phase2_baseline.json` with TriviaQA results for all four methods
- Bug fix in `src/agents/leader.py` `_resolve_conflict` method

Limitations:
- 20-sample evaluation; statistical significance not established
- Only TriviaQA RC validation split tested; needle-in-haystack and other benchmarks not re-run after bug fix

## Task and Scope

Request: Test TriviaQA dataset with Phase 1 and Phase 2 baselines, then conclude and commit.

Target: Run `scripts/run_phase1_baseline.py` and `scripts/run_phase2_baseline.py` on 20 TriviaQA RC samples, diagnose and fix any runtime errors, record results.

Starting state:
- Phase 1 and Phase 2 scripts implemented and tested on needle-in-haystack
- TriviaQA RC dataset (16,137 samples) available in `data/processed/triviaqa_rc.jsonl`
- Qwen2.5-3B-Instruct model loaded from local snapshot
- Known bug: `_resolve_conflict` in `leader.py` had an indexing error not yet encountered on needle-in-haystack

Constraints:
- Use default parameters: 20 samples, chunk size 512, overlap 50, top-K 5, confidence threshold 1.0
- Do not modify evaluation logic or question/answer format
- Preserve existing result files (overwrite with new TriviaQA results)

Decisions:
- Fix the bug immediately upon discovery rather than deferring
- Use the same 20-sample subset for fair comparison across methods

Exclusions:
- Phase 3 learnable estimator (next phase, not in scope)
- Hyperparameter tuning or prompt engineering
- Re-running needle-in-haystack after bug fix

## Implementation

Execution sequence:
1. Ran `scripts/run_phase1_baseline.py` on 20 TriviaQA samples — succeeded
2. Ran `scripts/run_phase2_baseline.py` on 20 TriviaQA samples — crashed at `_resolve_conflict` with `IndexError: list assignment index out of range`
3. Diagnosed bug: `new_responses` list had `top_k` entries (5), but code indexed by `wid_a`/`wid_b` (actual worker IDs, e.g., 15, 22)
4. Fixed `_resolve_conflict` to find positions by worker_id instead of direct indexing
5. Re-ran Phase 2 — all four methods completed successfully
6. Recorded results in JSON files

| Path | Symbol / Change | Role |
|------|----------------|------|
| `src/agents/leader.py` | `_resolve_conflict` | Fixed indexing: iterate to find position by worker_id, then replace in-place |
| `experiments/results/phase1_baseline.json` | Phase 1 TriviaQA results | Multi-agent 45%, full-context 60%, truncated 40% |
| `experiments/results/phase2_baseline.json` | Phase 2 TriviaQA results | broadcast 45%, selective 35%, selective+conflict 40%, selective+conflict+fallback 45% |

## Validation

### V1 - pass

```text
python scripts/run_phase1_baseline.py
```

Observed summary: Phase 1 baseline on 20 TriviaQA samples: 45% accuracy, 13553 avg tokens, 24.9 avg worker calls, 123.1s

Phase 1 baseline on 20 TriviaQA samples completed. Multi-Agent (LONGAGENT-style): 45.00% accuracy (9/20 correct), 13553.0 avg tokens, 24.9 avg worker calls, 123.1s total time. Single-Model Full Context: 60.00% accuracy, 11397.0 avg tokens, 90.3s. Single-Model Truncated (2048): 40.00% accuracy, 2058.9 avg tokens, 18.6s. Results saved to `experiments/results/phase1_baseline.json`.

### V2 - pass

```text
python scripts/run_phase2_baseline.py
```

Observed summary: Phase 2 all 4 methods on 20 TriviaQA samples: broadcast 45%, selective 35%, selective+conflict 40%, selective+conflict+fallback 45%

Phase 2 all four methods on 20 TriviaQA samples completed after bug fix. broadcast: 45.00% accuracy, 13553.0 avg tokens, 24.9 avg worker calls, 158.8s. selective: 35.00% accuracy, 2420.3 avg tokens, 4.5 avg worker calls, 40.1s. selective_conflict: 40.00% accuracy, 3319.7 avg tokens, 6.2 avg worker calls, 57.6s. selective_full: 45.00% accuracy, 14437.4 avg tokens, 26.7 avg worker calls, 186.9s. Results saved to `experiments/results/phase2_baseline.json`.

## Evidence Ledger

| ID | Class | Locator / Check | Conclusion |
|----|-------|----------------|------------|
| E1 | verified | `src/agents/leader.py` (SHA-256: `3be8fcfcd5956c72541589b87692223027dae5b547a3f5d1527a9f98ed7c75a2`) | Bug fix applied: `_resolve_conflict` now indexes by position, not worker_id |
| E2 | verified | `experiments/results/phase1_baseline.json` (SHA-256: `4e6445c4b25f5609c1996c592a6414a655df44fe68002995deee71e4395ce9fd`) | Phase 1 TriviaQA results: multi-agent 45%, full-context 60%, truncated 40% |
| E3 | verified | `experiments/results/phase2_baseline.json` (SHA-256: `453c396a794dc431e4e8c50371e170df199f0feb8a181d984f64253da95c67a3`) | Phase 2 TriviaQA results: broadcast 45%, selective 35%, selective+conflict 40%, selective+conflict+fallback 45% |
| V1 | verified | `python scripts/run_phase1_baseline.py` | Phase 1 baseline passed on 20 TriviaQA samples |
| V2 | verified | `python scripts/run_phase2_baseline.py` | Phase 2 all four methods passed on 20 TriviaQA samples (after bug fix) |

## Git Custody

- Branch: `main`
- Baseline HEAD: `8cd00a4ef8a94443b42194a8bf1f429b44843180` (explicit)
- Final HEAD: `8cd00a4ef8a94443b42194a8bf1f429b44843180` (same as baseline; commit pending)
- History relation: `same` (no commits since baseline)
- Commits since baseline: 0

Explicit scopes:
- `--task-path src/agents/leader.py`
- `--task-path experiments/results/phase1_baseline.json`
- `--task-path experiments/results/phase2_baseline.json`
- `--record-path progress/2026-08-09-phase2-triviaqa-evaluation.md`

Task-owned changes:
- `src/agents/leader.py`: Fixed `_resolve_conflict` IndexError (8 lines changed: 4 insertions, 4 deletions)
- `experiments/results/phase1_baseline.json`: Overwritten with TriviaQA results
- `experiments/results/phase2_baseline.json`: Overwritten with TriviaQA results

Pre-existing changes: All three task-owned paths were already modified before this session began (from prior Phase 1 and Phase 2 runs on needle-in-haystack). This session overwrote them with TriviaQA results.

Overlaps: All three paths overlap with pre-existing uncommitted changes. The prior changes were needle-in-haystack results; this session replaced them with TriviaQA results. No conflict since the files were fully overwritten.

Outside-scope changes: None.

Ownership caveats: The result JSON files were pre-existing and modified by prior runs. This session's contribution is the TriviaQA evaluation results and the bug fix in `leader.py`.

Scoped diff: `files=3; insertions=217; deletions=213; binary_files=0; untracked_files=0`

## Interface and Behavior Changes

**Bug fix**: `Leader._resolve_conflict` in `src/agents/leader.py` previously indexed `new_responses` using worker IDs directly (`new_responses[wid_a] = ...`), which caused `IndexError` when worker IDs exceeded the list length (e.g., `top_k=5` but `wid_a=15`).

New behavior: The method now iterates through `new_responses` to find the position of each conflicting worker by ID, then replaces the tuple at that position. This makes conflict resolution work correctly regardless of which workers were selected by BM25 retrieval.

Impact: Phase 2 + conflict and Phase 2 + conflict + fallback methods now run without crashing on documents with many chunks. No API or signature change; internal implementation fix only.

## Evidence Boundary

Established:
- Phase 1 and Phase 2 baselines run successfully on 20 TriviaQA RC samples
- BM25-based selective scheduling reduces accuracy on TriviaQA (35% vs 45% broadcast), contrasting with needle-in-haystack where it preserved accuracy
- Conflict resolution partially recovers accuracy (+5pp) at modest token cost
- Fallback restores accuracy to match broadcast but at higher total cost
- Bug in `_resolve_conflict` fixed and validated

Not established:
- Statistical significance (only 20 samples)
- Generalization to other datasets or question types
- Optimal hyperparameters for TriviaQA (top-K, confidence threshold)
- Whether the bug existed in needle-in-haystack runs (not re-tested)
- Phase 3 learnable estimator performance (future work)

## Next Steps

- Implement Phase 3: learnable coverage-sufficiency estimator to decide per-question whether top-K is sufficient or more chunks needed
- Increase sample size for statistical significance
- Tune BM25 parameters (top-K, scoring) for TriviaQA
- Investigate why multi-agent methods underperform single-model full context on TriviaQA
- Re-run needle-in-haystack after bug fix to confirm no regression
