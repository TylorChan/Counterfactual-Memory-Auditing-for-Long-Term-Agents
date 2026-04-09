# Current Plan: Counterfactual Memory Auditing Pipeline Hardening

Date: 2026-04-08

## Objective

Before any large rerun, make the counterfactual auditing pipeline structurally correct for all five agents so that:

1. provenance from answer-stage evidence back to original write(s) is reliable,
2. dominance is not a proxy for simple answer change,
3. metrics capture sensitivity/fragility rather than only net accuracy change,
4. all five agents can produce the same core artifacts with the same semantics,
5. reruns do not force midstream redesign of tracking, dominance, or aggregation.

The five target agents are:

- AnnaAgent
- SHARE
- MemoryOS
- LD-Agent
- THEANINE

## Current diagnosis

### A. The intervention is not the main weakness

Existing rollback/write-ablation traces already change behavior often enough to be useful. The current weak point is the measurement layer.

### B. Current after-CF accuracy is the wrong primary metric

The current export flow selects one max-influence CF answer per query and computes benchmark accuracy on that synthetic prediction set. This can hide strong effects by cancellation:

- some questions improve,
- some questions worsen,
- overall average accuracy stays flat.

This metric should become supplemental only.

### C. Current dominance is too blunt

Current influence:

- 1.0 * answer_changed
- 0.5 * abstention_flip
- 0.25 * retrieved_write_set_change
- 0.25 * prompt_write_set_change
- 0.25 * answer_distance

Current dominance:

- dominant iff influence_score > 0.75

Problem:

- for rollback over prompt-exposed writes, retrieved/prompt set change is often always true,
- therefore the absolute threshold almost collapses to “did the answer change?”,
- so the current 2x2 and Gini use a dominance signal that is too coarse.

### D. Provenance is incomplete and heterogeneous across agents

All five bridges currently emit `prompt_items` / `retrieved_items`, but the lineage quality is uneven:

- some agents map directly to raw ingress writes,
- some map only coarse summaries,
- some still leave answer-stage evidence in `bridge_items` as unmapped text blobs,
- some only partially track post-summary / post-refinement memory.

This makes the current 2x2 conservative and unstable, especially for systems with summary or knowledge layers.

## Audit findings from current repo state

The code audit on 2026-04-08 identified the following concrete implementation gaps that must be closed before any large rerun:

### Shared framework gaps

1. `longmemeval_audit.py` still stores flat item records and does not enforce a shared item-level provenance schema with:
   - `source_write_ids[]`
   - `source_session_ids[]`
   - `event_timestamps[]`
   - `memory_timestamps[]`
   - `source_form`

2. `longmemeval_counterfactual.py` still uses boolean retrieval/prompt set-change features and an absolute dominance threshold, which makes `dominant` too close to simple answer flip.

3. `scripts/aggregate_cf_query_matrix.py` still assumes binary dominance and does not surface `ambiguous` or `no_effect` query states.

4. `scripts/export_cf_eval_predictions.py` still selects one max-influence CF answer per query and should remain supplemental only, not the primary robustness readout.

### Agent-specific provenance gaps

#### MemoryOS

- short-term and page mapping are relatively strong,
- knowledge lineage is still text-keyed and not object-level,
- answer-stage knowledge evidence still depends on late lookup instead of first-class lineage attached at write time.

#### SHARE

- merge/update prompt explicitly handles duplicates and conflicts,
- but selected memory lineage is still effectively session-level,
- summary/update outputs do not yet preserve full unions of original source writes.

#### AnnaAgent

- unit-level tracking exists for some prompt items,
- native long-term retrieval still appears primarily as a blob,
- `anna_native_retrieval_blob` remains a major non-auditable source.

#### LD-Agent

- index-level lineage exists and is relatively strong,
- but answer-stage item records still flatten summary evidence too early,
- multi-source summary lineage is not preserved as first-class item metadata.

#### THEANINE

- `before_refinement` is partially auditable,
- `after_refinement` remains non-auditable,
- timeline refinement therefore does not yet participate in the causal provenance chain.

## Required output artifacts after the fix

Every agent rerun must emit the following with the same semantics:

### 1. Write records

One row per original ingress write with:

- `write_id`
- `session_id`
- `timestamp`
- `write_type`
- `content_text`
- `lineage_source_ids`
- `audit_eligible`

### 2. Query record

For every baseline query:

- `candidate_write_ids`
- `retrieved_write_ids`
- `selected_write_ids`
- `prompt_write_ids`
- `retrieved_items`
- `prompt_items`
- `bridge_items`
- `baseline_answer`

### 3. Item-level provenance for answer-stage evidence

Every `retrieved_item` / `prompt_item` must expose:

- `write_id` when one-to-one mapping is possible,
- or `source_write_ids[]` when the answer evidence comes from multiple writes,
- `stage`
- `score`
- `rank`
- `timestamp`
- `source_form`
- `audit_eligible`

### 4. CF run records

For every ablation run:

- answer delta metrics
- retrieval delta metrics
- prompt delta metrics
- dominance inputs
- target write provenance fields

### 5. Query summary

For every query:

- fragility metrics
- Gini metrics
- query-level dominance label
- query-level 2x2 label
- ETDL fields
- provenance coverage fields

## Phase 1: Replace the current primary metrics

### Goal

Make the main empirical story about hidden fragility and influence, not net after-CF accuracy.

### New primary metrics

For every agent, report:

1. `answer_flip_rate`
   - fraction of rollback runs where normalized answer changes

2. `query_fragility_rate`
   - fraction of queries with at least one rollback run that changes the answer

3. `abstention_flip_rate`
   - fraction of runs where abstention status changes

4. `mean_answer_distance`
   - mean normalized answer distance between baseline and CF answer

5. `max_answer_distance_per_query`
   - strongest per-query answer disruption

6. `mean_influence_per_query`
   - average write-ablation influence over tested writes

7. `rollback_gini`
   - per-query Gini over rollback influence scores

8. `provenance_coverage`
   - fraction of prompt/retrieved evidence items that map back to tracked write lineage

9. `query_level_2x2`
   - final professor-facing matrix, after dominance and provenance are repaired

10. `etdl_seconds`
   - temporal dependency estimate based on dominant or high-influence writes

### Supplemental metrics

Keep these as secondary:

- baseline accuracy
- after-CF exported accuracy
- per-category before/after accuracy

### Optional but strongly recommended

Add post-hoc answer confidence:

- force answer head to emit a scalar confidence,
- compute confidence change under write ablation,
- compute calibration metrics such as Brier or ECE if feasible.

This directly addresses the feedback that confidence may reveal write-level causal effects better than accuracy.

## Phase 2: Redefine influence and dominance

### Goal

Stop treating dominance as almost equivalent to answer flip.

### 2.1 Influence redesign

Replace boolean set-change terms with overlap-sensitive deltas:

- `retrieved_delta = 1 - Jaccard(R_base, R_cf)`
- `prompt_delta = 1 - Jaccard(P_base, P_cf)`

Keep:

- `answer_changed`
- `abstention_flip`
- `answer_distance`

Recommended influence form:

`influence = w1 * answer_changed + w2 * abstention_flip + w3 * answer_distance + w4 * retrieved_delta + w5 * prompt_delta`

Weights must be documented and kept fixed across agents.

### 2.2 Dominance redesign

Dominance should be query-relative, not absolute-threshold-only.

For each query:

1. compute influence for all tested writes,
2. rank writes by influence,
3. compute:
   - `top1`
   - `top2`
   - `top_share = top1 / sum(all influence)`
   - `margin = top1 - top2`

Then classify:

- `gold_dominant` if the top write is gold-support and sufficiently separated,
- `non_gold_dominant` if the top write is non-gold and sufficiently separated,
- `ambiguous` if top influences are too close,
- `no_effect` if all influences are near zero.

The query-level 2x2 should be computed from these final query labels, not from the current absolute threshold alone.

### 2.3 2x2 semantics

Final query-level 2x2:

- row: retrieved correctly / incorrectly
- col: causally dominant / non-dominant

Interpretation:

- retrieval correct = at least one gold-support write is exposed in baseline answer evidence
- dominant = a gold-support write is the clearly dominant causal write
- retrieved correctly + non-dominant = right memory present, not controlling
- retrieved incorrectly + dominant = wrong memory controls

`ambiguous` should be reported separately, not silently folded into non-dominant without visibility.

## Phase 3: Provenance and lineage hardening

### Global schema requirement

For every answer-stage evidence item, lineage must be explicit even after:

- summarization,
- memory update,
- refinement,
- page promotion,
- knowledge extraction,
- session merge.

Each evidence item must have either:

- `write_id`

or:

- `source_write_ids[]`

and ideally:

- `source_session_ids[]`
- `event_timestamps[]`
- `memory_timestamps[]`
- `source_form`

### Provenance acceptance criterion

For each agent, before large reruns:

- mapped prompt evidence coverage must be measured,
- mapped retrieved evidence coverage must be measured,
- unmapped bridge items must be enumerated by source,
- knowledge / summary / refinement layers cannot remain dominant hidden sources.

## Per-agent implementation plan

### MemoryOS

#### Current state

- short-term and page lineage are relatively strong,
- knowledge lineage exists through `knowledge_lineage_map`,
- but repaired summaries indicate knowledge provenance is not fully trusted downstream,
- query-level 2x2 is still conservative.

#### Required fixes

1. Make knowledge lineage first-class:
   - each user/assistant knowledge item should carry stable lineage metadata,
   - retrieval should recover object-level lineage, not depend only on normalized text lookup.

2. Distinguish:
   - original event timestamp,
   - knowledge write timestamp,
   - query time.

3. Report bucket-level provenance coverage:
   - short-term
   - page
   - retrieved_user_knowledge
   - retrieved_assistant_knowledge

4. Remove dependence on post-hoc repair for primary results.

#### Done only when

- a rerun can produce query-level 2x2 without `repair_memoryos_cf_provenance.py`.

### SHARE

#### Current state

- selected memories map back via session-level metadata,
- merge/update can compress several ingress events into one selected memory,
- lineage is too coarse for strong causal attribution.

#### Required fixes

1. Every extracted/updated memory item needs:
   - `source_write_ids[]`
   - `source_session_ids[]`

2. Merge/update should preserve lineage unions.

3. If conflict resolution occurs, preserve metadata such as:
   - `supersedes_write_ids[]`
   - `conflict_resolved`

4. Selection should expose lineage directly, not just map selected text back to a single session event.

#### Done only when

- selected summary memory can be traced to one or more original ingress writes without ambiguity hidden in `bridge_items`.

### AnnaAgent

#### Current state

- user/session units are reasonably structured,
- but native retrieved long-term text is exposed as a blob,
- fallback and short-term are partially mapped,
- answer evidence still contains non-auditable retrieval blobs.

#### Required fixes

1. Replace blob-level retrieved text attribution with unit-level mapping where possible.

2. For fallback memories, preserve:
   - `source_write_ids[]`
   - `session_id`
   - `turn_span`

3. Keep `real_time_context` non-auditable, but separate it cleanly from memory evidence.

4. Coverage report must distinguish:
   - retrieved long-term units
   - fallback prompt items
   - short-term prompt items
   - non-auditable retrieval blob

#### Done only when

- long-term retrieval no longer appears primarily as unmapped `bridge_items`.

### LD-Agent

#### Current state

- strongest lineage structure among the summary-heavy systems,
- uses `event_id_by_idx` and `long_term_lineage_by_idx`,
- but answer-stage attribution still compresses summary lineage too early.

#### Required fixes

1. Preserve summary lineage explicitly into retrieved/prompt items.

2. Expand long-term summary lineage from summary idx to original write ids.

3. Distinguish:
   - raw dialog evidence
   - summary evidence
   - lineage-expanded source writes

4. Ensure context and related memories are both auditable with the same schema.

#### Done only when

- every answer-stage summary item has a recoverable `source_write_ids[]`.

### THEANINE

#### Current state

- `before_refinement` can often be mapped,
- `after_refinement` is mostly non-auditable,
- therefore timeline refinement is not yet part of causal provenance.

#### Required fixes

1. Make refinement lineage explicit:
   - `parent_summary_node_ids[]`
   - `source_write_ids[]`

2. Mark refinement outputs as auditable when lineage is recoverable.

3. Keep both:
   - coarse session lineage,
   - fine write lineage.

4. Ensure answer-stage refined items are not hidden in `bridge_items`.

#### Done only when

- `after_refinement` contributes to tracked answer evidence rather than remaining opaque.

## Common implementation checklist before any large rerun

All five agents must pass the following:

1. `py_compile` on modified bridges and shared CF code
2. one-query smoke test
3. ten-query smoke test
4. provenance coverage report emitted
5. no silent `bridge_items` growth from major answer-stage evidence
6. query-level summary contains:
   - retrieval correctness
   - gold-support provenance
   - dominance label
   - ambiguity/no-effect label
   - Gini
   - fragility metrics
   - ETDL

## Rerun policy

### Stage 1

Rerun only:

- MemoryOS
- SHARE

on:

- at least 200 questions
- prompt-scope write ablation
- all prompt writes, or a high enough cap to avoid truncating gold-support evidence

### Stage 2

Once Stage 1 artifacts are stable, rerun:

- AnnaAgent
- LD-Agent
- THEANINE

with the same metric and provenance schema.

## Deliverables expected from the hardening phase

1. Updated shared CF code
2. Unified fragility metrics summary script
3. Unified provenance coverage summary script
4. Updated query-level 2x2 aggregation
5. Per-agent smoke-test report
6. Rerun-ready Slurm settings for all five agents

## Final target for the next 20 days

This project should not end as:

- a benchmark add-on,
- a small-sample curiosity,
- or a claim that only relies on after-CF accuracy drop.

The final result should be a unified empirical and methodological argument that:

**long-term memory systems have a hidden causal structure that standard benchmark accuracy does not measure.**

That argument should be supported across all five agents:

- AnnaAgent
- SHARE
- MemoryOS
- LD-Agent
- THEANINE

### Final scientific claim

The final presentation and report should make the following four claims:

1. **Benchmark accuracy misses write-level fragility.**
   - Even when average LongMemEval accuracy changes little after intervention, individual answers can still be highly sensitive to the removal of specific writes.

2. **Answers are often causally concentrated in a small subset of writes.**
   - This is captured with Gini over per-query write-ablation influence scores.

3. **Correct retrieval does not guarantee correct causal control.**
   - A system can fetch gold-support memory but still be driven by the wrong write.
   - This is captured with the professor-facing query-level 2x2:
     - retrieved correctly / incorrectly
     - causally dominant / non-dominant

4. **Agent outputs can remain sensitive to older memory writes farther back than benchmark accuracy alone would suggest.**
   - This is captured with temporal dependency length and a survival curve over write age.

### Final empirical package

For all five agents, the final package should include:

1. **Expanded-sample baseline results**
   - baseline LongMemEval scores on a larger slice than the current 50-question subset

2. **Expanded-sample counterfactual fragility results**
   - answer flip rate
   - query fragility rate
   - abstention flip rate
   - mean answer distance
   - mean / max influence per query

3. **Expanded-sample influence concentration results**
   - Gini distributions
   - mean / median Gini
   - agent comparison plots

4. **Expanded-sample query-level retrieval–dominance mismatch results**
   - professor-facing 2x2 for each agent
   - off-diagonal counts as the main diagnostic story
   - ambiguity/no-effect counts reported separately

5. **Expanded-sample temporal dependency results**
   - ETDL summary statistics
   - survival curve over dependency length / memory age
   - note whether any long-tail or non-monotonic “zombie memory” behavior appears

6. **Case studies**
   - at least one strong MemoryOS case
   - at least one non-MemoryOS case
   - each case should show:
     - question
     - gold answer
     - answer-stage evidence
     - original write provenance
     - baseline answer
     - counterfactual answer

### Final methodological position

The final framing should be:

- not “we found a new benchmark,”
- not “we caused benchmark accuracy to drop,”
- but:

**we introduce the first practical framework for measuring causal responsibility in long-term memory systems, and we show that benchmark accuracy alone misses fragility, concentration, retrieval–dominance mismatch, and long-range memory sensitivity.**

### Final minimum acceptable outcome

At minimum, by the final presentation, the project should deliver:

1. all five agents running the same provenance-aware audit pipeline,
2. all five agents producing the same core artifacts:
   - baseline audit
   - CF runs
   - CF query summaries
   - Gini
   - query-level 2x2
   - ETDL / temporal dependency outputs
3. a larger rerun than the current 50-question subset,
4. a stable and defensible narrative that does not depend on after-CF accuracy as the main effect.

### Final stretch outcome

If runtime and engineering time permit, the strongest version of the project is:

1. all five agents rerun on a substantially expanded sample,
2. 5-agent comparisons for:
   - fragility
   - Gini
   - query-level 2x2
   - temporal survival
3. confidence-sensitive robustness as an extra readout,
4. a final talk that argues:
   - standard LTM evaluation is blind to hidden causal structure,
   - correct retrieval is necessary but not sufficient,
   - write-level causal auditing reveals system-specific memory pathologies that standard QA scores miss.

### Practical execution priority

Given the remaining time, execution should prioritize:

1. make all five agents smoke-test clean,
2. make all five agents rerun-ready on MSI,
3. produce 5-agent large-sample core metrics,
4. use accuracy only as a baseline reference,
5. treat fragility, Gini, 2x2, and temporal dependency as the real final results.

## Explicit non-goals for this phase

Not doing yet:

- attention tracing
- pairwise write interaction / entanglement experiments
- policy sensitivity experiments
- fluency evaluation
- broader benchmark expansion beyond LongMemEval and current LoCoMo path

Those can be revisited after the metric/dominance/provenance chain is stable.

## Definition of success

This hardening phase is complete only when:

1. all five agents can emit the same provenance-aware audit schema,
2. dominance is no longer a near-alias of answer flip,
3. 2x2 results do not depend on ad hoc repair scripts,
4. provenance coverage is explicitly measured and acceptable,
5. reruns can proceed without expecting another redesign of the tracking chain.
