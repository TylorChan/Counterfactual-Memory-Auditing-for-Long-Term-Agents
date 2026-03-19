# Counterfactual Wrapper Design

## Goal
This wrapper is designed to produce the three outputs requested by the professor without contaminating baseline traces:

1. Distribution of influences across memories or write/update events
2. A 2x2 confusion matrix over retrieval correctness and causal dominance
3. Temporal dependency length and survival-curve-ready outputs for zombie-memory analysis

The wrapper is enabled per bridge with a dedicated CLI flag and writes to separate CF trace files.

## Best-Practice Principles

1. Baseline must remain clean.
   - Native memory pipeline is left unchanged.
   - Unified QA head remains the same as the baseline answer head.
   - CF traces are written to separate files.

2. Counterfactual execution must happen at true write ingress.
   - Interventions are applied before native writes/updates enter the agent memory pipeline.
   - The rest of the native pipeline runs unchanged.
   - The wrapper never mutates final evidence rows directly.

3. Core interventions must be deterministic.
   - The wrapper never asks an LLM to invent interventions.
   - The wrapper only executes pre-specified rollback and time-shift rules.

4. Time-shift interventions must be temporally coherent.
   - Content is unchanged.
   - Only timestamps or effective write order are changed.
   - Query time is fixed.

5. The same counterfactual contract is used across all five agents.
   - Same rule names.
   - Same influence formula.
   - Same dominance threshold.
   - Same retrieval-correctness contract.

## True Write-Time Replay Architecture

Each agent exposes a native write ingress unit. The wrapper enumerates those units, applies an intervention to one unit, then replays the original pipeline.

Execution contract:

1. Build baseline write events from the sample.
2. Run baseline replay with no intervention.
3. Build counterfactual specs over baseline write IDs.
4. For each spec:
   - modify or skip the target write event before it enters memory
   - replay the native pipeline end-to-end
   - record answer, retrieved IDs, prompt IDs, and derived influence
5. Aggregate per-query metrics.

This is true write-time replay. It is not answer-time evidence patching.

## Agent-Native Write Ingress Units

The write unit is defined at the earliest practical point where the bridge can intercept input before it is committed to memory, without changing native memory logic.

- MemoryOS: QA-pair ingress into `memo.add_memory(...)`
- SHARE: per-session memory-update ingress before `update_share_memory(...)`
- LD-Agent: dialogue-turn ingress before appending into short-term memory and later retrieval/flush
- Anna: `MemoryUnit` ingress before retrieval over long-term / short-term memory views
- THEANINE: history-session ingress before episode construction, summarization, and graph linking

This means the unit is agent-native, but not necessarily identical across agents. That is acceptable because the professor asked about influences across memories and updates; the wrapper measures the dominant native write/update unit for each system.

## Retrieval Alignment Contract

The baseline replay produces a query record aligned to ingress write IDs:

- `candidate_write_ids`
- `retrieved_write_ids`
- `selected_write_ids`
- `prompt_write_ids`

These IDs refer to ingress write units, not final rendered evidence rows.

Where a downstream memory object cannot be cleanly mapped back to an ingress unit, it is recorded as a non-audit bridge item and excluded from the dominant-write universe.

## Interventions

### Rollback
`rollback_skip(write_id)`

Semantics:
- remove the target write before it enters the native memory pipeline
- replay all later writes unchanged
- rerun retrieval and answer generation

Use:
- influence distribution
- causal dominance
- ETDL
- confusion matrix

### Time Shift
`time_shift(write_id, rule_id)`

Semantics:
- keep content fixed
- modify only the target write's timestamp or effective ordering before replay
- rerun the native memory pipeline and answer generation

Use:
- recency sensitivity
- temporal dependency
- zombie-memory resurfacing

## Rule Library by Question Type

### temporal-reasoning
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`
- `timeshift_cross_query_boundary`

### knowledge-update
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`
- `timeshift_cross_query_boundary`

### multi-session
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`

### single-session-assistant
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`

### single-session-preference
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`

### single-session-user
- `rollback_skip`
- `timeshift_promote_before_query`
- `timeshift_demote_far_past`

## Retrieval Correctness

Retrieval correctness is defined against the baseline replay state.

A write is treated as gold-supporting if:
- its `lineage_source_ids` intersects the sample's `answer_session_ids`, or
- its `session_id` matches one of the sample's `answer_session_ids`

This keeps the first implementation deterministic and auditable.

## Influence Score

For each intervention target `w_i`:

`Influence(w_i) = 1.0 * answer_changed + 0.5 * abstention_flip + 0.25 * retrieved_write_set_change + 0.25 * prompt_write_set_change + 0.25 * answer_distance`

Where:
- `answer_changed` is exact normalized answer inequality
- `abstention_flip` captures transitions into or out of `I don't know.`
- `retrieved_write_set_change` compares baseline vs CF retrieved ingress write IDs
- `prompt_write_set_change` compares baseline vs CF prompt ingress write IDs
- `answer_distance` is lexical distance between baseline and CF answers

## Dominance

A write is dominant if:
- `Influence(w_i) > dominance_threshold`

Default threshold:
- `0.75`

## Metrics Produced

### 1. Influence Distribution
Per query:
- one rollback influence score per target write
- Gini over rollback influence scores

Output field:
- `rollback_gini`

### 2. 2x2 Confusion Matrix
Over rollback runs on retrieved writes:
- retrieved correct + dominant
- retrieved correct + non-dominant
- retrieved incorrect + dominant
- retrieved incorrect + non-dominant

This is the main off-diagonal analysis.

### 3. Temporal Dependency Length
Per query:
- compute age of each rollback target relative to query time
- keep ages only for dominant writes
- `ETDL = max(age_seconds)`

Across queries:
- saved `etdl_seconds` values are survival-curve ready
- run-level traces also support age-bucket influence analysis for zombie memories

## Output Files

Baseline remains unchanged.

CF wrapper writes separate files derived from the bridge trace path:
- `*.cf_runs.jsonl`
- `*.cf_queries.jsonl`

`cf_runs` stores one row per intervention.
`cf_queries` stores one row per question with aggregated metrics.

## CLI Contract

All five bridges expose the same CF arguments:
- `--enable-cf-wrapper`
- `--cf-target-scope prompt|retrieved|candidate`
- `--cf-max-writes N`
- `--cf-dominance-threshold FLOAT`

Recommended default for the first paper-quality run:
- `--enable-cf-wrapper`
- `--cf-target-scope prompt`
- `--cf-max-writes 0`
- `--cf-dominance-threshold 0.75`
