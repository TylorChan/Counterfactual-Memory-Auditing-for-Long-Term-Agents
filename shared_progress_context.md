## When agent reads this file, they SHOULD
This file is the cross-machine progress context for this repo. Use it so Codex on Google VM, laptop, and MSI can quickly align on current status, decisions, blockers, and next actions.

1. Identify which machine section to update based on the user's request.
2. Prefer maintaining one concise summary block per machine section instead of appending many historical update blocks.
3. Keep the machine section concise and factual; do not add workflow headers such as `High-level progress`, `Completed`, `In progress`, `Blockers`, or `Next steps` unless the user explicitly asks for that structure.
4. Summarize only facts that are verifiable from repo files, logs, commands, or explicit user confirmation.
5. Preserve discussion outcomes that materially affect experiment design, interpretation, fairness, runtime decisions, or evaluation protocol.
6. Remove duplicate or stale points inside the same machine section while preserving the newest factual wording.
7. Keep exactly one `(latest)` label in the whole file:
   - Add `(latest)` to the section you updated.
   - Remove `(latest)` from all other section titles.
8. Include one rough UTC timestamp for the updated section when helpful; exact minute precision is optional.
9. If a value is uncertain, mark it as `TBD` instead of guessing.

## Google VM progress summary

- Update time (UTC): 2026-03-05
- High-level progress:
  - Cross-agent bridge alignment is mostly complete; final benchmarking is in the execution/consolidation phase.
  - 50-question plan is partially complete: Anna and SHARE have completed runs, while MemoryOS and LD-Agent still need finalization for a fair 4-agent table.
- Conversation highlights so far:
  - We established a cross-machine continuity workflow (shared context + structured handoff/changelog) so VM/MSI/Mac Codex sessions can resume with minimal context loss.
  - We standardized bridge run/evaluation commands to machine-agnostic pathing (`REPO_ROOT`) instead of hard-coded absolute paths.
  - We fixed evaluation reliability issues (`OPENAI_API_KEY` loading and OpenAI/httpx compatibility handling in `evaluate_qa.py`).
  - We diagnosed SHARE memory bottlenecks and moved to no-cap memory setting as the primary comparison direction.
  - We diagnosed and fixed LD-Agent startup failure caused by `chromadb` vs NumPy 2.x mismatch by pinning NumPy 1.26.4.
  - We aligned a Git sync workflow for multi-machine work (`push` from source machine, `pull/reset` on target machine).
- Current risks:
  - Final report is blocked until MemoryOS and LD-Agent produce finalized comparable outputs.
  - Runtime artifacts and partial/resume outputs still need consolidation into one clean final result set.
- Next actions:
  - Finish remaining agent runs and finalize one prediction file per agent for the same 50-question subset.
  - Run unified evaluation and produce a single high-level comparison table (accuracy + per-task + runtime).
  - Keep this file updated at each machine handoff with only high-level, decision-relevant deltas.

## My laptop progress summary (latest)

- Update time (UTC): 2026-03-14 15:05:12
- MacBook work shifted from ChatGPT-Web/THEANINE exploration to fairness control for the LongMemEval agent comparison.
- Added a shared unified factual QA head in `longmemeval_unified_answer.py` and rewired all 5 bridges (`Anna`, `SHARE`, `LD-Agent`, `MemoryOS`, `THEANINE`) to use the same final answer prompt while leaving each agent’s native memory write/retrieve pipeline unchanged.
- This design decision was explicit: normalize only the final answer head first, and do **not** repair temporal reasoning yet, so later counterfactual results reflect memory/retrieval behavior rather than dialogue-style prompt mismatch.
- Local inspection of bridge code and existing trace files found baseline `retrieval-correct but influence-wrong` cases before any counterfactual intervention; two concrete examples discussed were `SHARE` question `c8c3f81d` (`Nike` evidence present but model abstains) and `MemoryOS` question `852ce960` (correct `$400,000` memory present but stale `$350,000` memory dominates).
- The 5 modified bridges plus the shared prompt helper passed `py_compile`; dry-run smoke checks succeeded for the patched `MemoryOS` and `THEANINE` bridges.
- For MSI execution, a new Slurm array script `run_agents_array_unifiedqa.slurm` was created and shell-validated. It now runs the unified-QA baselines as 9 tasks: `THEANINE`, `SHARE`, `MemoryOS`, and `LD-Agent` are split into two 25-question shards each, while `Anna` stays whole.
- The Slurm array is configured as `0-8%6`, ordered longest-first to reduce makespan, and output filenames now carry a default `MM_DD` suffix plus shard tag (for example `..._03_14_p1.jsonl`) so new baseline files do not collide with older outputs.
- The MSI launch plan now assumes six OpenAI keys/projects are available via `.env`: `OPENAI_API_KEY`, `OPENAI_API_KEY_1`, `...`, `OPENAI_API_KEY_5`, with the first 6 concurrent shards mapped one-to-one to distinct keys.
- Next expected machine handoff: upload the updated code to MSI, run the new unified-QA baseline array there, then use those new baseline traces/results as the reference point before implementing the counterfactual replay wrapper.

## MSI progress summary

- Update time (UTC): 2026-03-12 (approx)
- MSI is the main execution and consolidation machine for LongMemEval comparison across Anna, SHARE, MemoryOS, LD-Agent, and THEANINE.
- Repo commands were standardized to `REPO_ROOT=\"$(git rev-parse --show-toplevel)\"` so run/eval paths stay machine-agnostic across MSI, laptop, and VM.
- The 50-example subset `/users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json` was checked to be usable for the shared benchmark and near-balanced across the 6 question types.
- `evaluate_qa.py` now auto-loads `.env`, writes logs into `LongMemEval/eval_result/`, and is invoked with repo-root-relative paths.
- Memory-alignment decisions used for the benchmark: MemoryOS uses `retrieval_queue_capacity=7` with `--reset-mode reinit`; LD-Agent defaults to `--no-force-flush-before-answer`; Anna uses corrected role mapping with `full_tertiary_init` off and `need-check` off by default; SHARE keeps its original memory module shape but uses the no-cap setting as the main comparison condition.
- SHARE and LD-Agent both got `0/9` on the temporal questions in this 50-example subset. Trace analysis suggests this is mainly a memory/retrieval-structure issue rather than a simple bridge bug: SHARE often retrieves semantically related but temporally unusable memories and abstains with `I don't know`, while LD-Agent often reaches answer generation with only `0-1` related memories, which is usually not enough for multi-event temporal reasoning.
- The temporal `0/9` result should not be overinterpreted as true accuracy `0`; the subset only contains 9 temporal questions, so the sample is small, but the failure pattern was consistent enough to indicate a real structural weakness.
- THEANINE was integrated as an additional agent after the original 4-agent setup. Local upstream was patched to support dynamic session counts, multi-digit memory keys such as `s10-m1`, and to filter empty summary lines before embedding.
- THEANINE full-history on MSI reached 47 successful predictions in `preds_theanine_s_50_fullhist.jsonl`; the run then failed on question 48 with OpenAI `insufficient_quota`, and a separate resume Slurm was prepared to finish the last 3 questions without overwriting the first 47 outputs.
- MSI Git sync is currently easiest via SSH; HTTPS push failed because GitHub no longer accepts password authentication for Git operations.
- `/users/9/chen7751/.codex/memories` is empty, so this file is the main maintained summary of MSI-side context.
