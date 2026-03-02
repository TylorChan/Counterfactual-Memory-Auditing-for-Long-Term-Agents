# LD-Agent x LongMemEval Bridge

This folder provides an isolated runner to evaluate `LD-Agent` on `LongMemEval` without modifying upstream repositories.

## Files

- `run_infer.py`: replays LongMemEval sessions into LD-Agent memory modules and writes predictions JSONL.

## Conda environment (`ld-lme`)

```bash
conda create -n ld-lme python=3.10 -y
conda activate ld-lme
pip install --upgrade pip
pip install openai==1.12.0 httpx==0.27.2 chromadb==0.4.22 spacy==3.7.4 tqdm pillow
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
```

## Prerequisites

1. Local repos exist:
- `/users/9/chen7751/csci8980/LD-Agent`
- `/users/9/chen7751/csci8980/LongMemEval`

2. Python environment includes at least:
- `openai`
- `chromadb`
- `spacy`
- `en_core_web_sm` model
- `tqdm`

Install spaCy model if missing:

```bash
python -m spacy download en_core_web_sm
```

3. API key:

```bash
export OPENAI_API_KEY=...
```

`run_infer.py` auto-loads `.env` from:
- current working directory
- workspace root (`/users/9/chen7751/csci8980/.env`)
- bridge folder (`ldagent_longmemeval_bridge/.env`)

By default, existing environment variables are not overridden.

## Quick start (smoke test)

```bash
cd /users/9/chen7751/csci8980
python ldagent_longmemeval_bridge/run_infer.py \
  --ld-agent-dir /users/9/chen7751/csci8980/LD-Agent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50_smoke.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50_smoke.trace.jsonl \
  --llm-model gpt-4o-mini \
  --limit 3
```

Dry-run (no API calls):

```bash
cd /users/9/chen7751/csci8980
python ldagent_longmemeval_bridge/run_infer.py \
  --ld-agent-dir /users/9/chen7751/csci8980/LD-Agent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50_dryrun.jsonl \
  --limit 3 \
  --dry-run
```

## Main run (50-question subset)

```bash
cd /users/9/chen7751/csci8980
python ldagent_longmemeval_bridge/run_infer.py \
  --ld-agent-dir /users/9/chen7751/csci8980/LD-Agent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50.trace.jsonl \
  --session-gap-seconds 600 \
  --llm-model gpt-4o-mini \
  --fail-fast
```

## Paper-aligned run (50-question subset)

This command aligns more closely with LD-Agent paper choices for memory behavior:
- `--session-gap-seconds 600` (paper setting)
- `--ori-mem-query`
- `--dist-thres 0.5`

```bash
cd /users/9/chen7751/csci8980
python ldagent_longmemeval_bridge/run_infer.py \
  --ld-agent-dir /users/9/chen7751/csci8980/LD-Agent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50_paper.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50_paper.trace.jsonl \
  --llm-model gpt-4o-mini \
  --session-gap-seconds 600 \
  --no-force-flush-before-answer \
  --ori-mem-query \
  --dist-thres 0.5 \
  --fail-fast
```

Evaluate with LongMemEval script:

```bash
cd /users/9/chen7751/csci8980/LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o \
  /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_50.jsonl \
  /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json
```

## Full run (LongMemEval-S, 500 questions)

```bash
cd /users/9/chen7751/csci8980
python ldagent_longmemeval_bridge/run_infer.py \
  --ld-agent-dir /users/9/chen7751/csci8980/LD-Agent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_full.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_ldagent_s_full.trace.jsonl \
  --llm-model gpt-4o-mini \
  --fail-fast
```

## Notes

- State is reset per question by creating fresh LD-Agent memory modules, preventing cross-question leakage.
- Sessions are replayed in chronological order by default; use `--preserve-session-order` to disable sorting.
- Final query prepends `question_date` by default; use `--omit-question-date` to disable.
- Persona extraction is enabled by default; use `--disable-persona-update` for speed ablation.
- `--session-gap-seconds` controls when short-term cache is summarized into long-term memory. Use `600` to align with LD-Agent paper setting; default is `3600` (matches current upstream code).
- `--no-force-flush-before-answer` is the default (repo-like behavior): after ingest, final QA runs without extra forced short-term -> long-term flush.
  - Use `--force-flush-before-answer` only as ablation.
- `--dist-thres` is only behavior-critical when `--ori-mem-query` is enabled.
  - If `--ori-mem-query` is off, retrieval is primarily driven by noun-overlap/time-decay scoring in upstream `EventMemory.relevance_retrieve`.
- Output schema matches LongMemEval evaluation input (`question_id`, `hypothesis`).
