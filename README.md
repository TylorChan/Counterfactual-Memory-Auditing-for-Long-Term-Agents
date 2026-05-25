# Counterfactual Memory Auditing for Long-Term Agents

This project evaluates long-term memory agents with counterfactual memory auditing. The core idea is simple: first run a memory agent normally on LongMemEval, trace which original memory writes reach the answer stage, remove one traced evidence write, replay the same question, and compare the baseline and counterfactual answers. This lets us test not only whether an agent is accurate, but also whether the right memory was retrieved and causally used.

## Project Report

Read the report on GitHub Pages: <https://tylorchan.github.io/Counterfactual-Memory-Auditing-for-Long-Term-Agents/>

Direct PDF link: [Beyond Accuracy: Counterfactual Auditing of Long-Term Agent Memory](docs/project_report.pdf)

## What This Code Measures

- **Baseline accuracy**: LongMemEval accuracy before any counterfactual intervention.
- **Counterfactual accuracy**: LongMemEval accuracy after removing one traced evidence write.
- **Answer flip rate**: how often the final answer changes after a traced write is removed.
- **Abstention flip rate**: how often the answer changes into or out of `I don't know`.
- **Counterfactual Gini**: whether answer influence is concentrated in a few memory writes.
- **ETDL / temporal horizon**: how old the influential memory writes are relative to the query time.
- **Retrieval--dominance matrix**: whether the gold memory was retrieved and whether it actually controlled the answer.
- **Provenance coverage**: how much answer-stage evidence can be traced back to original writes.

## Repository Layout

- `LongMemEval/`: datasets, predictions, traces, logs, and campaign outputs.
- `anna_longmemeval_bridge/`: bridge for AnnaAgent.
- `share_longmemeval_bridge/`: bridge for SHARE.
- `memos_longmemeval_bridge/`: bridge for MemoryOS.
- `ldagent_longmemeval_bridge/`: bridge for LD-Agent.
- `theanine_longmemeval_bridge/`: bridge for THEANINE.
- `mem0_longmemeval_bridge/`: bridge for the local Mem0 setup.
- `longmemeval_audit.py`: shared provenance and audit record utilities.
- `longmemeval_counterfactual.py`: shared counterfactual comparison and metric logic.
- `longmemeval_unified_answer.py`: unified LongMemEval answer prompt used across bridges.
- `openai_prompt_cache.py`: OpenAI prompt-cache wrapper and token logging.
- `scripts/`: smoke tests, MSI submitters, shard runners, and aggregation scripts.
- `slurms/run_*_unifiedqa_cf_5x10.slurm`: MSI Slurm job files for the main agents.
- `papers/`: local copies of the agent papers.
- `report/`: ACL-style final report source and figures.

The upstream agent repositories are included as local folders: `AnnaAgent/`, `SHARE/`, `MemoryOS/`, `LD-Agent/`, and `Theanine/`.

## Setup

Create a `.env` file in the repo root with OpenAI keys. For MSI shard runs, multiple keys can be used:

```bash
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_API_KEY_1=<OPENAI_API_KEY_1>
OPENAI_API_KEY_2=<OPENAI_API_KEY_2>
...
OPENAI_API_KEY_10=<OPENAI_API_KEY_10>
```

Do not commit `.env` or generated runtime folders. The run scripts load `.env` automatically.

Each agent uses its own Conda environment:

- `anna-lme`
- `share-lme`
- `memos-lme`
- `ld-lme`
- `theanine-lme`
- `mem0-lme`

The shard scripts activate the correct environment automatically.

## Quick Smoke Test

Run the smallest local wiring test for the five main agents:

```bash
bash scripts/run_smoke_ltm.sh --preset wiring
```

Run a slightly larger local logic test:

```bash
bash scripts/run_smoke_ltm.sh --preset logic
```

Mem0 has a separate smoke path:

```bash
bash scripts/run_mem0_smoke_local.sh
```

These smoke tests are mainly for checking environment setup, tracing, counterfactual replay, and metric aggregation before launching expensive MSI jobs.

## Run the 100-Question MSI Campaign

The main evaluation uses MSI Slurm jobs and shards the dataset across parallel jobs. By default, the submitter uses:

- `LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json`
- `TOTAL_QUESTIONS=100`
- `SHARDS=15`
- `KEY_SLOTS=10`
- `LLM_MODEL=gpt-4o-mini`
- prompt-cache logging enabled

Submit the five main agents:

```bash
bash scripts/submit_unifiedqa_cf_campaign.sh
```

Submit Mem0 separately:

```bash
bash scripts/submit_mem0_unifiedqa_cf_campaign.sh
```

Useful overrides:

```bash
TOTAL_QUESTIONS=50 SHARDS=10 KEY_SLOTS=5 bash scripts/submit_unifiedqa_cf_campaign.sh
DATA_FILE=/path/to/data.json SHARDS=15 bash scripts/submit_unifiedqa_cf_campaign.sh
```

Each campaign writes outputs under:

```bash
LongMemEval/<MM_DD_HH_MM>/
```

The `logs/` subdirectory contains Slurm stdout/stderr and prompt-cache token logs.

## Aggregate Metrics

Aggregate counterfactual query metrics from a campaign directory:

```bash
python scripts/aggregate_cf_metrics.py \
  LongMemEval/<RUN_DIR>/*.trace.cf_queries.jsonl \
  --out-json LongMemEval/<RUN_DIR>/cf_metrics_summary.json
```

Build a retrieval--dominance matrix for a given agent output:

```bash
python scripts/aggregate_cf_query_matrix.py \
  --cf-queries LongMemEval/<RUN_DIR>/<AGENT>.trace.cf_queries.jsonl \
  --cf-runs LongMemEval/<RUN_DIR>/<AGENT>.trace.cf_runs.jsonl \
  --out-json LongMemEval/<RUN_DIR>/<AGENT>.query_matrix.json
```

The main output file types are:

- `*.jsonl`: baseline predictions.
- `*.trace.jsonl`: baseline trace with write IDs and provenance.
- `*.trace.audit_queries.jsonl`: baseline query-level audit records.
- `*.trace.audit_writes.jsonl`: baseline write-level audit records.
- `*.trace.cf_runs.jsonl`: per-counterfactual-run records.
- `*.trace.cf_queries.jsonl`: per-query counterfactual summary records.
- `*.query_matrix.json`: retrieval--dominance matrix output.
