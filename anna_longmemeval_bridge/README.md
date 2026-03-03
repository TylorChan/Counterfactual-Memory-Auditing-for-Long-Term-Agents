# AnnaAgent x LongMemEval Bridge

`run_infer.py` adapts AnnaAgent's multi-session retrieval path to LongMemEval:
- uses AnnaAgent `querier.is_need` + `querier.query` for long-term retrieval,
- keeps a short-term context window,
- generates the final answer with the same LLM.

## Environment

```bash
conda create -n anna-lme python=3.10 -y
conda activate anna-lme
pip install --upgrade pip
pip install -r /users/9/chen7751/csci8980/anna_longmemeval_bridge/requirements.txt
```

Set key (or put it in `/users/9/chen7751/csci8980/.env`):

```bash
export OPENAI_API_KEY=...
```

## Run (50-question subset)

```bash
cd /users/9/chen7751/csci8980
python anna_longmemeval_bridge/run_infer.py \
  --anna-agent-dir /users/9/chen7751/csci8980/AnnaAgent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_anna_s_50.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_anna_s_50.trace.jsonl \
  --llm-model gpt-4o-mini \
  --fail-fast
```

Dry run:

```bash
python anna_longmemeval_bridge/run_infer.py \
  --anna-agent-dir /users/9/chen7751/csci8980/AnnaAgent \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_anna_s_50_dryrun.jsonl \
  --limit 3 \
  --dry-run
```

## Evaluate with LongMemEval script

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/LongMemEval/src/evaluation"
python evaluate_qa.py gpt-4o \
  "$REPO_ROOT/LongMemEval/preds_anna_s_50.jsonl" \
  "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json"
```

## Notes

- Output schema is LongMemEval-compatible (`question_id`, `hypothesis`).
- By default, sessions are replayed in chronological order.
- The bridge now uses a tertiary-style split:
  - long-term memory: earlier sessions (fed to `querier.query`)
  - short-term memory: latest `--short-term-sessions` sessions (default `1`)
  - real-time memory: current question turn
- Role mapping is aligned with Anna semantics for retrieval:
  - `Seeker` = LongMemEval `user`
  - `Counselor` = LongMemEval `assistant`
- Full tertiary initialization is disabled by default to reduce bridge-side synthetic bias.
  - Use `--enable-full-tertiary-init` to enable style / scales / situation / status / complaint-chain initialization.
  - `--disable-full-tertiary-init` is kept as a compatibility alias.
- If the split would leave long-term empty, the bridge backfills long-term with all sessions by default for robustness; use `--strict-tertiary-split` to disable that fallback.
- Need-check gate is disabled by default (`is_need` off): bridge always runs Anna `query` for long-term retrieval.
  - Use `--enable-need-check` to restore Anna `is_need` gate.
  - `--disable-need-check` is kept as a compatibility alias.
