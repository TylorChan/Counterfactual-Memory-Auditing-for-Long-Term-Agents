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


- Update time (UTC): 2026-03-06 17:33:30
- High-level progress: MacBook-side automation and THEANINE integration both advanced materially; ChatGPT Web project automation is now working end-to-end for smoke cases, and THEANINE bridge now supports full-history LongMemEval adaptation.
- Completed:
  - Built `/Users/daqingchen/csci8980/chatgpt_web_eval` Playwright automation for ChatGPT Web using an existing Chrome CDP session.
  - Verified project creation flow on ChatGPT Web: expand `Projects` -> `New project` -> set `Project-only` memory -> reopen project settings -> write instructions -> click `Save`.
  - Added single-question runner and batch runner with tqdm-like progress formatting under `/Users/daqingchen/csci8980/chatgpt_web_eval`.
  - Updated ChatGPT Web protocol so each session starts with `For this conversation only, assume the current date is <session_date>. Use this date when answering in this chat.`
  - Updated project instructions to constrain verbosity globally for both replay chats and final QA.
  - Forked / cloned `/Users/daqingchen/csci8980/Theanine` and built `/Users/daqingchen/csci8980/theanine_longmemeval_bridge`.
  - Patched local THEANINE upstream to support dynamic session counts instead of the original fixed `4 history + 1 current` design.
  - Patched local THEANINE memory-key parsing to support multi-digit session ids such as `s10-m1`.
  - Verified full-history dry-run for THEANINE: first question `gpt4_9a159967` now uses `46` history sessions plus QA session `47` in `/Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_fullhist_dryrun.trace.jsonl`.
- In progress:
  - Real full-history THEANINE smoke run in `theanine-lme` environment is running / being validated for runtime viability.
  - ChatGPT Web batch automation exists, but full 50-question product run is not practically viable under ChatGPT message caps.
- Blockers / risks:
  - ChatGPT Plus product-side `GPT-5.3 Instant` message limits are a hard bottleneck for full 50-question LongMemEval replay; exact count computation showed every question in `longmemeval_s_cleaned_50.json` exceeds the 160 messages / 3h cap under the current replay protocol.
  - ChatGPT Web product baseline is therefore suitable only for smoke / small-sample product evaluation, not full 50-question main-table evaluation.
  - THEANINE full-history adaptation is now code-complete locally, but runtime cost may be very high because graph construction now scales over all history sessions per question.
- Next steps:
  - Let the THEANINE full-history smoke finish, then decide whether full 50-question execution is tractable on available budget/time.
  - If THEANINE full-history is too slow, run an explicit truncated-history ablation with `--history-sessions N` and label it as such.
  - Keep ChatGPT Web automation for product-side smoke / supplementary baseline only, not main agent comparison.

## MSI progress summary (latest)

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
