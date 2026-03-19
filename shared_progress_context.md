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

## My laptop progress summary

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

## MSI progress summary (latest)

- Update time (UTC): 2026-03-19 (approx)
- MSI is the main execution and consolidation machine for the unified-QA baseline and CF-only rollback analysis across Anna, SHARE, MemoryOS, LD-Agent, and THEANINE.
- The unified factual QA head is now the baseline comparison setting for all 5 bridges; the native memory pipelines remain unchanged, and bridge traces were extended with audit-aligned write/query records for later CF replay.
- CF design was settled around true write-time replay: rollback/time-shift rules target ingress write events before they enter each agent’s native memory pipeline; CF outputs are written separately as `*.cf_runs.jsonl` and `*.cf_queries.jsonl` so baseline traces stay clean.
- Full baseline outputs for the main comparison run are the `03_14_22_46` files in `LongMemEval/`; Anna is `full`, SHARE/MemoryOS/LD-Agent are `p1+p2`, and THEANINE has `48/50` usable baseline examples.
- The strongest currently usable CF results are in `/users/9/chen7751/csci8980/cf_compare_results/`, which contains:
  - `memoryos_original_accuracy.txt` / `memoryos_afterCF_accuracy.txt`
  - `share_original_accuracy.txt` / `share_afterCF_accuracy.txt`
  - original baseline traces and CF traces for MemoryOS and SHARE
  - merged eval result files for baseline vs CF-exported predictions
  - `gini_comparison_memoryos_share.png` for the presentation Gini slide
- Current verified accuracy comparison from `cf_compare_results`:
  - MemoryOS baseline `0.50`; after rollback-CF export `0.50`
  - SHARE baseline `0.20`; after rollback-CF export `0.20`
  - These CF exports are not “improved models”; they use the highest-influence rollback answer per query as an eval proxy, so unchanged total accuracy does not mean rollback had no effect.
- Current verified rollback sensitivity summary from `cf_compare_results`:
  - MemoryOS: `150` rollback runs over `50` queries; `58/150` runs changed the answer; `30/50` queries have at least one dominant write; mean Gini `0.1143`; median Gini `0.0026`; non-null ETDL on `25/50` queries with mean `3.96` days and max `40.7` days.
  - SHARE: `95` rollback runs over `50` queries; `9/95` runs changed the answer; `7/50` queries have at least one dominant write; mean Gini `0.0241`; median Gini `0.0`; non-null ETDL on `6/50` queries with mean `8.78` days and max `29.12` days.
- Interpreting the current CF results: rollback is already useful as a diagnostic intervention, especially for MemoryOS, but not as an accuracy-improving intervention. MemoryOS shows stronger answer sensitivity and higher influence concentration than SHARE.
- The first professor-facing metric is presentation-ready now: Gini over rollback influence scores can be shown for MemoryOS vs SHARE. The ETDL story is presentation-ready as an initial rollback-based temporal sensitivity result, strongest for MemoryOS.
- The second professor-facing metric is not presentation-ready as a rigorous result yet: in the current MemoryOS and SHARE CF summaries, `gold_support_write_ids` and `baseline_retrieval_correct` are still empty/false for all queries, so the 2x2 retrieval-correctness off-diagonal story cannot yet be defended as complete.
- Presentation guidance agreed in discussion:
  - Do not sell CF as an accuracy-improving method; present it as a diagnostic intervention for measuring memory influence.
  - Show the before/after accuracy table, then a Gini slide, then a MemoryOS temporal sensitivity/ETDL slide or a strong MemoryOS case study.
  - Use a simple influence-score explanation on the Gini slide: rollback a write, compare answer / abstention / retrieval-prompt changes / answer-text distance, then compute Gini across multiple rollback targets for the query.
  - Use a plain-language Gini title such as “Do A Few Memories Dominate the Answer?” rather than a heavily technical title.
- Rollback-only CF infrastructure has dedicated Slurm scripts for MemoryOS, SHARE, LD-Agent, and THEANINE. MemoryOS and SHARE completed successfully enough to analyze; LD-Agent and THEANINE hit malformed-JSON OpenAI 400s in long runs.
- Those LD-Agent/THEANINE long-run 400s were patched at the current code level by adding prompt cleaning and targeted retries around the relevant OpenAI request paths. New rerun Slurms were created:
  - `run_ldagent_cf_only_rollback_rerun_s2.slurm` reruns only the failed LD-Agent `s2`
  - `run_theanine_cf_only_rollback_rerun_full.slurm` reruns THEANINE fully from patched source
- `/users/9/chen7751/.codex/memories` is empty, so this file remains the main maintained MSI-side summary.
