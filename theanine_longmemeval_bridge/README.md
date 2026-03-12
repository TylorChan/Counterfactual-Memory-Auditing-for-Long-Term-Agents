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
- `"$REPO_ROOT/Theanine"`

The patch makes THEANINE handle:
- dynamic session counts,
- multi-digit session ids such as `s10-m1`,
- a final QA session at `history_session_count + 1`.

This is no longer the original fixed `4 history + 1 current` upstream behavior.

## Environment

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
conda create -n theanine-lme python=3.10 -y
conda activate theanine-lme
pip install --upgrade pip
pip install -r "$REPO_ROOT/theanine_longmemeval_bridge/requirements.txt"
```

Set key (or put it in `"$REPO_ROOT/.env"`):

```bash
export OPENAI_API_KEY=...
```

The bridge writes `"$REPO_ROOT/Theanine/conf.d/config.yaml"` at runtime because upstream code expects that file.

## Smoke test

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python "$REPO_ROOT/theanine_longmemeval_bridge/run_infer.py" \
  --theanine-dir "$REPO_ROOT/Theanine" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_smoke.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_smoke.trace.jsonl" \
  --llm-model gpt-4o-mini \
  --limit 1 \
  --fail-fast
```

## Main run (50-question subset, full history)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python "$REPO_ROOT/theanine_longmemeval_bridge/run_infer.py" \
  --theanine-dir "$REPO_ROOT/Theanine" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50.trace.jsonl" \
  --llm-model gpt-4o-mini
```

## Optional truncated-history run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python "$REPO_ROOT/theanine_longmemeval_bridge/run_infer.py" \
  --theanine-dir "$REPO_ROOT/Theanine" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_hist10.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_hist10.trace.jsonl" \
  --llm-model gpt-4o-mini \
  --history-sessions 10
```

## Dry run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python "$REPO_ROOT/theanine_longmemeval_bridge/run_infer.py" \
  --theanine-dir "$REPO_ROOT/Theanine" \
  --longmemeval-file "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json" \
  --out-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_dryrun.jsonl" \
  --trace-jsonl "$REPO_ROOT/LongMemEval/preds_theanine_s_50_dryrun.trace.jsonl" \
  --limit 3 \
  --dry-run
```

## Evaluate with LongMemEval script

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python "$REPO_ROOT/LongMemEval/src/evaluation/evaluate_qa.py" gpt-4o \
  "$REPO_ROOT/LongMemEval/preds_theanine_s_50.jsonl" \
  "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json"
```
