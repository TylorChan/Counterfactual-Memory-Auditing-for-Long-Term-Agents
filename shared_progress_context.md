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
- High-level progress: bridge alignment + evaluation compatibility work is mostly done; experiment execution is still the bottleneck.
- Objective status: 50-question fair comparison across 4 agents is not complete yet (`Anna` done, `SHARE` done, `MemoryOS` partial, `LD-Agent` pending).
- Agent run status:
  - `Anna`: `LongMemEval/preds_anna_s_50.jsonl` = 50/50.
  - `SHARE (max10 old run)`: `LongMemEval/preds_share_s_50_max10MemoryCap.jsonl` = 50/50.
  - `SHARE (nocap current run)`: `LongMemEval/preds_share_s_50_nocap.jsonl` = 50/50.
  - `MemoryOS base file`: `LongMemEval/preds_memoryos_s_50.jsonl` = 18/50.
  - `MemoryOS resume file`: `LongMemEval/preds_memoryos_s_50.resume.jsonl` = 31 lines (not merged to a final 50-line file yet).
  - `LD-Agent`: `LongMemEval/preds_ldagent_s_50.jsonl` missing.
- Evaluation snapshot (gpt-4o autoeval on available files):
  - `Anna`: `15/50 = 0.30` from `preds_anna_s_50.jsonl.eval-results-gpt-4o`.
  - `SHARE max10`: `7/50 = 0.14` from `preds_share_s_50_max10MemoryCap.jsonl.eval-results-gpt-4o`.
  - `SHARE nocap`: `10/50 = 0.20` from `preds_share_s_50_nocap.jsonl.eval-results-gpt-4o`.
  - Observed delta: removing SHARE memory cap improved accuracy by `+0.06` absolute on this subset (`0.14 -> 0.20`).
- Important implemented decisions (for cross-machine continuity):
  - SHARE bridge default memory cap is disabled (`--memory-max-items` default `0`) in `share_longmemeval_bridge/run_infer.py`.
  - Bridge evaluation commands were migrated to machine-independent `REPO_ROOT` style in bridge READMEs.
  - `LongMemEval/src/evaluation/evaluate_qa.py` now auto-loads `.env` key and includes compatibility fallback for `proxies`/httpx client init.
- Runtime/ops facts:
  - `LongMemEval/memoryos_progress_record.txt` still shows checkpoint `18/50` with PID `191860`; PID is no longer running.
  - Current working tree contains many runtime-output diffs; treat them as experiment artifacts until final selection/cleanup.
- Blockers / risks:
  - No finalized MemoryOS 50-line merged output yet.
  - LD-Agent run has not produced the 50-question predictions file yet.
  - Final 4-agent table (accuracy + per-task + runtime) still pending because 2 agents are incomplete.
- Next steps (handoff-ready):
  - Complete MemoryOS run and produce one final deduplicated/merged `preds_memoryos_s_50*.jsonl`.
  - Run LD-Agent 50-question bridge inference and generate `preds_ldagent_s_50.jsonl`.
  - Evaluate all final 4 files with `evaluate_qa.py`, then build one consolidated comparison table.

## My laptop progress summary

## MSI progress summary
