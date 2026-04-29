# Official Mem0 LongMemEval CF Bridge

This bridge targets the `mem0ai/memory-benchmarks` LongMemEval pipeline rather than the lightweight `mem0ai` SDK bridge.

Protocol:

1. Import official benchmark code from `external/memory-benchmarks` pinned by default to `f75666d33ef560f0f196746e0e16c515d17e6856`.
2. Use official LongMemEval session ordering, user/assistant pair chunking, answer-generation prompt, Mem0 REST/Cloud client, and async LLM client.
3. Treat each official ingestion pair as one auditable write with a stable `write_id`.
4. For baseline, ingest all writes, search Mem0 with the official query path, generate the answer with the official prompt, and emit unified CF audit artifacts.
5. For CF, replay the same official pipeline under a fresh `user_id`, but skip the target write before it enters Mem0. This is true write-time rollback and does not modify Mem0 internals.
6. Do not judge every CF replay. Export one baseline answer and all CF answers/traces; use `scripts/export_cf_eval_predictions.py` to select max-influence CF answers for final after-CF accuracy evaluation.

Primary outputs match the existing five-agent schema:

- `preds_mem0_official_*.jsonl`
- `preds_mem0_official_*.trace.jsonl`
- `*.trace.audit_writes.jsonl`
- `*.trace.audit_queries.jsonl`
- `*.trace.cf_runs.jsonl`
- `*.trace.cf_queries.jsonl`

Cloud mode is the closest available path to Mem0's reported platform LongMemEval result. OSS mode is useful for plumbing tests, but should not be described as the platform 94% setup.
