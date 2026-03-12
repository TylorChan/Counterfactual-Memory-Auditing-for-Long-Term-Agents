---
name: summarize-context
description: Update and maintain `shared_progress_context.md` as a concise cross-machine summary. Use when the user asks to summarize recent verified progress, merge old and new machine-specific notes, consolidate conversation context into one maintained section, or refresh the MSI/laptop/VM section with a rough UTC timestamp.
---

# Summarize Context

Read the relevant section in `shared_progress_context.md` and update only the machine section requested by the user unless they explicitly ask for more.

Keep the update compact. Prefer one maintained summary block per machine section instead of adding many historical mini-blocks when the user asks for consolidation.

Only include facts that are verifiable from:
- files in the repo
- logs or output files
- commands you ran
- explicit user confirmation in the current conversation

Treat the recent conversation itself as an important source of summary content when it contains experimentally relevant conclusions. Preserve discussion outcomes that materially affect experiment design, interpretation, or execution strategy.

If there is a local memory store available, read only the entries relevant to the requested machine and merge any non-duplicative factual points. If no such memory exists, do not invent it.

When updating the section:
- Keep exactly one `(latest)` label in the whole file
- Move `(latest)` to the section you updated
- Remove `(latest)` from other section titles
- Include one rough UTC timestamp line such as `2026-03-12 (approx)` when exact minute precision is unnecessary
- Write concise factual bullets only
- Prefer summarizing important findings from the conversation, especially:
  - experimentally observed phenomena
  - comparisons the user explicitly cared about
  - conclusions that affect fairness of agent comparison
  - decisions that changed run configuration, evaluation protocol, or interpretation
- Do not add workflow headers such as `High-level progress`, `Completed`, `In progress`, `Blockers`, `Next steps` unless the user explicitly asks for that structure
- Do not include minor implementation or environment bugs when they are not relevant to agent design, memory behavior, evaluation validity, or experiment conclusions
- Remove stale duplicate points and keep the newest factual wording

For MSI updates in this repo, treat these items as especially important when they are supported by files or logs:
- bridge alignment changes
- Slurm script status
- eval workflow changes
- runtime or resume state for long runs
- major failure analyses (for example temporal reasoning diagnostics)
- important discussion conclusions from the conversation, including user-noticed phenomena that affected how results were interpreted
- Git/auth state if it affects repo sync

After editing, quickly reread the updated section and check:
- the `(latest)` marker is unique
- the section is concise
- no speculative claims were added
