# MemoryOS x LongMemEval Bridge

This directory contains an isolated integration runner to evaluate `MemoryOS` on `LongMemEval` without modifying either upstream repository.

## Why this layout

- Keeps upstream repos clean (`MemoryOS/`, `LongMemEval/`).
- Makes experiments reproducible and easy to version.
- Avoids merge/update conflicts when pulling latest upstream changes.

## Files

- `run_infer.py`: replay LongMemEval sessions into MemoryOS and write predictions JSONL.

## Prerequisites

1. Both repos exist locally:
- `<repo-root>/MemoryOS`
- `<repo-root>/LongMemEval`

2. Python env has MemoryOS dependencies installed (at least those used by `memoryos-pypi`).

3. API key is set:

```bash
export OPENAI_API_KEY=...
```

The runner also auto-loads `.env` if present (search order):
- current working directory: `.env`
- workspace root (one level above this folder): `.env`
- bridge folder: `memos_longmemeval_bridge/.env`

Existing exported env vars are kept (the `.env` loader does not override them).

## Conda Environment Setup (Recommended)

Use a dedicated conda env for this bridge (CPU workflow):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
conda create -n memos-lme python=3.10 -y
conda activate memos-lme
python -m pip install --upgrade pip setuptools wheel

# CPU FAISS for MSI CPU partitions (recommended for this bridge)
conda install -y -c conda-forge faiss-cpu

# Upstream requirements include faiss-gpu; filter it out for CPU env
grep -v '^faiss-gpu' "$REPO_ROOT/MemoryOS/memoryos-pypi/requirements.txt" \
  > /tmp/memoryos_requirements_cpu.txt
python -m pip install -r /tmp/memoryos_requirements_cpu.txt

# Bridge runtime dependency
python -m pip install tqdm
```

Quick verify:

```bash
python -c "import faiss, openai, sentence_transformers, tqdm; print('memos-lme ready')"
```

## Quick start (smoke test)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python memos_longmemeval_bridge/run_infer.py \
  --memoryos-dir "$REPO_ROOT/MemoryOS" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_smoke.jsonl" \
  --limit 3 \
  --llm-model gpt-4o-mini
```

Dry-run (no API call, validate pipeline/output format only):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python memos_longmemeval_bridge/run_infer.py \
  --memoryos-dir "$REPO_ROOT/MemoryOS" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_dryrun.jsonl" \
  --limit 3 \
  --dry-run
```

Then evaluate:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python "$REPO_ROOT/LongMemEval/src/evaluation/evaluate_qa.py" gpt-4o \
  "$REPO_ROOT/LongMemEval/preds_memoryos_s_smoke.jsonl" \
  "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned.json"
```

## Main run (50-question stratified subset)
The full longmemeval_s is too big. It will take me 7 days to feed all the history and get the answers of 500 questions. 
So I pick 50 question (multi-session: 12; temporal-reasoning: 12; knowledge-update: 7; single-session-user: 6; single-session-assistant: 6 ;single-session-preference: 3 ;abstention(_abs): 4)
Run the fixed 50-question subset in `longmemeval_s_cleaned_50.json`:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python memos_longmemeval_bridge/run_infer.py \
  --memoryos-dir "$REPO_ROOT/MemoryOS" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_50.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_50.trace.jsonl" \
  --reset-mode reinit \
  --llm-model gpt-4o-mini \
  --fail-fast
```

Evaluate 50-question outputs:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python "$REPO_ROOT/LongMemEval/src/evaluation/evaluate_qa.py" gpt-4o \
  "$REPO_ROOT/LongMemEval/preds_memoryos_s_50.resumed_merged.jsonl" \
  "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json"
```

## Full run (LongMemEval-S, 500 questions)

Run all samples in `longmemeval_s_cleaned.json` (slow and expensive):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python memos_longmemeval_bridge/run_infer.py \
  --memoryos-dir "$REPO_ROOT/MemoryOS" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_full.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_memoryos_s_full.trace.jsonl" \
  --llm-model gpt-4o-mini \
  --fail-fast
```

## Notes

- The bridge resets MemoryOS state before each question to avoid cross-sample leakage.
- `--reset-mode reinit` (default) rebuilds a fresh official `Memoryos` instance for each question, minimizing bridge-side state mutation and maximizing repo alignment.
  - Use `--reset-mode manual` only if you need speed and accept bridge-side internal reset logic.
- Default `mid_term_heat_threshold` is `5` (paper/repo behavior).
- Default `retrieval_queue_capacity` is `7` (paper/repo setting).
- Output JSONL format matches LongMemEval evaluation requirements: one line per question with:
  - `question_id`
  - `hypothesis`
- By default, sessions are replayed in chronological order using `haystack_dates` (important for online memory fairness).
  - Use `--preserve-session-order` to disable sorting.
- By default, final query is prefixed with `Current date: ...` using `question_date`.
  - Use `--omit-question-date` to disable.
- By default, the bridge does not override MemoryOS LLM decoding params.
  - Use `--response-temperature` and/or `--response-max-tokens` only when you need an explicit override.
- Progress is displayed with `tqdm`. MemoryOS internal logs are muted by default; use `--verbose-memoryos` to enable them.
