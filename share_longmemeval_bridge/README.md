# SHARE x LongMemEval Bridge

`run_infer.py` replays each LongMemEval sample, then runs a SHARE-style pipeline aligned with the paper method:
1) session memory extraction, 2) memory update, 3) memory selection, 4) answer generation.

## Environment

```bash
conda create -n share-lme python=3.10 -y
conda activate share-lme
pip install --upgrade pip
pip install -r /users/9/chen7751/csci8980/share_longmemeval_bridge/requirements.txt
```

Set key (or put it in `/users/9/chen7751/csci8980/.env`):

```bash
export OPENAI_API_KEY=...
```

## Run (50-question subset)

```bash
cd /users/9/chen7751/csci8980
python share_longmemeval_bridge/run_infer.py \
  --share-dir /users/9/chen7751/csci8980/SHARE \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_share_s_50.jsonl \
  --trace-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_share_s_50.trace.jsonl \
  --llm-model gpt-4o-mini \
  --strict-selection-mode qa \
  --fail-fast
```

Dry run:

```bash
python share_longmemeval_bridge/run_infer.py \
  --share-dir /users/9/chen7751/csci8980/SHARE \
  --longmemeval-file /users/9/chen7751/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json \
  --out-jsonl /users/9/chen7751/csci8980/LongMemEval/preds_share_s_50_dryrun.jsonl \
  --limit 3 \
  --dry-run
```

## Evaluate with LongMemEval script

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/LongMemEval/src/evaluation"
python evaluate_qa.py gpt-4o \
  "$REPO_ROOT/LongMemEval/preds_share_s_50.jsonl" \
  "$REPO_ROOT/LongMemEval/data/longmemeval_s_cleaned_50.json"
```

## Notes

- Output schema is LongMemEval-compatible (`question_id`, `hypothesis`).
- By default, sessions are replayed in chronological order.
- Strict EPISODE memory mode is enabled by default: official `update_task` prompt+parser flow is used for extraction/update/selection.
- Strict selection mode defaults to `qa` for LongMemEval fairness while keeping strict EPISODE memory extraction/update intact.
  - Use `--strict-selection-mode qa` for benchmark-aligned QA evidence selection.
  - Use `--strict-selection-mode dialogue` for original EPISODE next-turn selection behavior (ablation/reference).
- Strict mode now uses JSON-constrained I/O by default for extraction/update/selection and falls back to legacy strict text parsing only on failures.
  - This reduces format drift while keeping the same SHARE memory schema and update buckets.
  - Use `--disable-strict-json-io` to force legacy strict text-only parsing.
- In strict mode, the bridge now auto-falls back to the JSON pipeline if strict parser output is empty for a session/update.
  - Check trace fields `strict_extract_fallbacks` / `strict_update_fallbacks` when diagnosing quality.
  - Also check `strict_extract_text_fallbacks` / `strict_update_text_fallbacks` / `strict_select_text_fallbacks` to track JSON->text fallback frequency.
- Use `--disable-strict-episode-memory` to fall back to the lightweight generic pipeline.
- By default, answer generation uses best-effort answering and only abstains when evidence is truly missing.
  - Use `--force-abstain-when-uncertain` to restore strict "I don't know" behavior.
- By default, `mutual` events are migrated into `shared` memory after each update (closer to EPISODE behavior).
- By default, memory selection candidates exclude `mutual`; use `--include-mutual-in-candidates` for ablation.
- Use `--retain-mutual-memory` if you want to keep a separate `mutual` bucket in traces.
- Use `--omit-question-date` to disable date prefix in final query.
