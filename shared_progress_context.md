## When agent reads this file, they SHOULD
This file is the cross-machine progress context for this repo. Use it so Codex on Google VM, laptop, and MSI can quickly align on current status, decisions, blockers, and next actions.

1. Identify which machine section to update based on the user's request.
2. Prefer maintaining one concise summary block per machine section instead of appending many historical update blocks.
3. Keep the machine section concise and factual; do not add workflow headers such as `High-level progress`, `Completed`, `In progress`, `Blockers`, or `Next steps` unless the user explicitly asks for that structure.
4. Summarize only facts that are verifiable from repo files, logs, commands, or explicit user confirmation.
5. Preserve discussion outcomes that materially affect experiment design, interpretation, fairness, runtime decisions, or evaluation protocol.
6. Remove duplicate or stale points inside the same machine section while preserving the newest factual wording.
7. Keep exactly one latest-marker in the whole file:
   - Add the latest-marker to the section you updated.
   - Remove the latest-marker from all other section titles.
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

- Update time (UTC): 2026-04-09 (approx)
- MacBook work moved from presentation-only interpretation back to pipeline hardening so the final reruns can support paper-level claims rather than small-sample slide results.
- `currentplan.md` now fixes the final target as a 5-agent causal-memory analysis: all five agents should report baseline benchmark results, fragility metrics, Gini/influence concentration, a query-level retrieval–dominance 2×2, provenance coverage, ETDL/temporal dependency outputs, and case studies on an expanded sample larger than the current 50-question slice.
- The final story is now explicit in the local plan: benchmark accuracy misses hidden causal structure; the real results should show write-level fragility, concentrated influence, retrieval–dominance mismatch, and temporal dependency beyond what standard LongMemEval accuracy reveals.
- Shared audit/CF code was hardened on the laptop so that baseline traces and CF summaries use a common provenance-aware schema (`audit_v2` / `cf_audit_v2`) with item-level `source_write_ids`, query-relative dominance labels (`gold_dominant`, `non_gold_dominant`, `ambiguous`, `no_effect`), provenance coverage fields, and ETDL outputs.
- The shared aggregation layer now has direct outputs for the professor-facing query-level 2×2 and for cross-agent metric summaries, including ETDL survival-curve points in `scripts/aggregate_cf_metrics.py`.
- A MacBook smoke script `scripts/run_smoke_ltm.sh` plus conda bootstrap helpers were added so each agent can be baseline-tested, CF-tested, and matrix-tested locally before any MSI rerun.
- Verified local smoke run completed end-to-end for all 5 agents with tag `smoke_0409_005451` / `cf_smoke_0409_005451`; each agent produced baseline audit output, CF run/query JSONL, and a query-matrix JSON under `LongMemEval/`.
- Smoke-driven code fixes on the laptop hardened the runtime path for all five agents: Anna replay now uses the correct retriever path, SHARE lineage normalization no longer collides with its local helper, and shared scripts no longer depend on hard-coded local conda paths.
- Verified from the smoke artifacts: `Anna`, `SHARE`, `MemoryOS`, and `LD-Agent` all produced baseline + CF + matrix files with the new schema; `MemoryOS`, `SHARE`, and `LD-Agent` showed full item coverage on the smoke example, while `Anna` still had `retrieved_item_coverage = 0.0` on that one example even though the end-to-end run succeeded.
- `THEANINE` completed the full smoke run, but the first smoke artifacts exposed a provenance gap: the baseline audit for that example placed all answer-stage evidence in `bridge_items` with zero retrieved/prompt coverage even though CF still produced a dominant write and non-null ETDL.
- After the smoke run, the laptop code patched `THEANINE` provenance again by mapping `before_refinement` text through bracketed summary segments instead of whole-string matching; offline reconstruction on the same smoke trace now recovers `retrieved_items = 6`, `prompt_items = 6`, `bridge_items = 0`, and a gold-support retrieval intersection on that example. This means future reruns should use the patched code, not the stale smoke artifact, for `THEANINE`.
- The smoke artifacts also showed that Gini values are `0.0` under `cf_max_writes=1`; this is expected and only confirms the code path. Substantive Gini claims still require larger reruns with more rollback targets per query.
- ETDL/temporal dependency is now structurally available in the code path, but the large rerun must be used to validate it across agents because many smoke summaries had null ETDL on the single tested example.
- Practical next step from the laptop side: sync the patched code to MSI and rerun a larger slice (targeting roughly 100–150 questions, and higher if runtime permits) across all five agents with the hardened pipeline, then aggregate the final figures from fragility, Gini, 2×2, provenance coverage, and temporal dependency instead of relying on after-CF average accuracy.

## MSI progress summary (latest)

- Update time (UTC): 2026-04-23 (approx)
- Older MSI context from `2026-03-19` is preserved below unchanged; this new block records only the newer MSI-relevant conclusions discussed and verified after that summary.
- `longmemeval_counterfactual.py` was corrected so the query-level 2x2 retrieval row no longer uses broad answer-stage exposure as `baseline_retrieval_correct`; the row now uses agent-specific primary retrieval items, while the old broad signal is retained separately as `baseline_exposure_correct`.
- For `MemoryOS`, the primary retrieval row now means only `retrieved_page` / `memoryos_page`; `retrieved_user_knowledge` and `retrieved_assistant_knowledge` remain audited evidence but no longer count as “the right memory was fetched” for the professor-facing 2x2.
- Offline recomputation on the existing `LongMemEval/04_09_18_11` `MemoryOS` artifacts changed the 2x2 from a collapsed top row to: `retrieved_correct_dominant = 15`, `retrieved_correct_non_dominant = 23`, `retrieved_incorrect_dominant = 1`, `retrieved_incorrect_non_dominant = 11`. The dominant/non-dominant column totals stayed `16/34`, so the change was only in the retrieval-row semantics.
- Concrete `MemoryOS` example for the 2x2 fix: query `2698e78f` (“How often do I see my therapist, Dr. Smith?”) had a correct baseline answer, `old_exposure_correct = True`, and `new_primary_retrieval_correct = False`. The overlap with gold support writes came only from `retrieved_user_knowledge` / `retrieved_assistant_knowledge`; `retrieved_page` overlap was `0`. This is the clearest current example of why the previous retrieval-correctness definition was too broad.
- Metric interpretation was clarified for the MSI-facing analysis plan: high baseline accuracy can coexist with high fragility. On `04_09_18_11`, `MemoryOS` remains the strongest currently discussed baseline system while also showing `query_fragility = 33/50`, which is being treated as evidence of stronger memory grounding rather than simple weakness.
- Raw cross-agent influence comparisons are now treated as IDK-confounded unless accompanied by baseline abstention context. On `04_09_18_11`, baseline `I don't know` rates were: `MemoryOS 0.38`, `THEANINE 0.46`, `SHARE 0.86`, `LD-Agent 0.94`; this means low raw influence for `SHARE` or `LD-Agent` cannot be interpreted as robustness by itself.
- `findings.md` now records two MSI-relevant interpretation conclusions: the current setup measures agent-specific memory pipelines under a shared LongMemEval QA readout rather than native end-to-end prompting, and `MemoryOS` currently exhibits the notable “high baseline accuracy + high counterfactual fragility” pattern.
- `Anna` MSI rerun state: `LongMemEval/04_19_17_05` is not valid for CF analysis because the CF path produced empty `cf_queries` / `cf_runs` and polluted `preds` with `ERROR: name 'client' is not defined`. `anna_longmemeval_bridge/run_infer.py` has since been patched (`llm=llm`), so future MSI Anna-only reruns should use the patched code.

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
