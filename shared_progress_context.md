## When agent reads this file, they SHOULD
This file is the cross-machine progress context for this repo. Use it so Codex on Google VM, laptop, and MSI can quickly align on current status, decisions, blockers, and next actions.

1. Identify which machine section to update based on the user's request.
2. Append one new update block with a UTC timestamp instead of overwriting older notes.
3. Keep each update block concise and structured with:
   - High-level progress
   - Completed
   - In progress
   - Blockers / risks
   - Next steps
4. Remove duplicate points inside the same machine section while preserving the most recent factual version.
5. Keep exactly one `(latest)` label in the whole file:
   - Add `(latest)` to the section you updated.
   - Remove `(latest)` from all other section titles.
6. Only record facts that are verifiable from files, logs, commands, or explicit user confirmation.
7. If a value is uncertain, mark it as `TBD` instead of guessing.

## Google VM progress summary (latest)

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

## MSI progress summary
