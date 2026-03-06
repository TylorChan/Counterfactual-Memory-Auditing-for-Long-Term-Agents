# THEANINE x LongMemEval Bridge

`run_infer.py` adapts THEANINE to LongMemEval by replaying each sample's full haystack history as THEANINE history sessions, then appending one final QA session.

Bridge protocol:
- convert each LongMemEval `haystack_session` into one THEANINE history session,
- summarize all history sessions,
- construct the memory graph over all summarized history sessions,
- append one final QA session containing only the final query,
- run THEANINE retrieval / refinement / response generation on that QA session.

## Current behavior

By default, the bridge replays **all** history sessions in each LongMemEval sample.

Optional truncation is still available:
- `--history-sessions 0` means replay all haystack sessions,
- `--history-sessions N` means replay only the first `N` history sessions after ordering.

The trace file records:
- how many history sessions were requested,
- how many history sessions were actually used,
- selected session ids and dates,
- the final QA session number,
- omitted `answer_session_ids` if truncation is enabled.

## Upstream patch in this workspace

This bridge now relies on a local upstream patch under:
- `/Users/daqingchen/csci8980/Theanine`

The patch makes THEANINE handle:
- dynamic session counts,
- multi-digit session ids such as `s10-m1`,
- a final QA session at `history_session_count + 1`.

This is no longer the original fixed `4 history + 1 current` upstream behavior.

## Environment

```bash
conda create -n theanine-lme python=3.10 -y
conda activate theanine-lme
pip install --upgrade pip
pip install -r /Users/daqingchen/csci8980/theanine_longmemeval_bridge/requirements.txt
```

Set key (or put it in `/Users/daqingchen/csci8980/.env`):

```bash
export OPENAI_API_KEY=...
```

The bridge writes `/Users/daqingchen/csci8980/Theanine/conf.d/config.yaml` at runtime because upstream code expects that file.

## Smoke test

```bash
conda run -n theanine-lme python /Users/daqingchen/csci8980/theanine_longmemeval_bridge/run_infer.py \
  --theanine-dir /Users/daqingchen/csci8980/Theanine \
  --longmemeval-file /Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_smoke.jsonl \
  --trace-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_smoke.trace.jsonl \
  --llm-model gpt-4o-mini \
  --limit 1 \
  --fail-fast
```

## Main run (50-question subset, full history)

```bash
conda run -n theanine-lme python /Users/daqingchen/csci8980/theanine_longmemeval_bridge/run_infer.py \
  --theanine-dir /Users/daqingchen/csci8980/Theanine \
  --longmemeval-file /Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50.jsonl \
  --trace-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50.trace.jsonl \
  --llm-model gpt-4o-mini
```

## Optional truncated-history run

```bash
conda run -n theanine-lme python /Users/daqingchen/csci8980/theanine_longmemeval_bridge/run_infer.py \
  --theanine-dir /Users/daqingchen/csci8980/Theanine \
  --longmemeval-file /Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_hist10.jsonl \
  --trace-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_hist10.trace.jsonl \
  --llm-model gpt-4o-mini \
  --history-sessions 10
```

## Dry run

```bash
conda run -n theanine-lme python /Users/daqingchen/csci8980/theanine_longmemeval_bridge/run_infer.py \
  --theanine-dir /Users/daqingchen/csci8980/Theanine \
  --longmemeval-file /Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_dryrun.jsonl \
  --trace-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50_dryrun.trace.jsonl \
  --limit 3 \
  --dry-run
```

## Evaluate with LongMemEval script

```bash
cd /Users/daqingchen/csci8980/LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o \
  /Users/daqingchen/csci8980/LongMemEval/preds_theanine_s_50.jsonl \
  /Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json
```
