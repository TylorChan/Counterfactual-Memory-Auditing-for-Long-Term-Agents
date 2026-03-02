# Overall timeline
- Search on existing, advanced memory management systems/agents. (2 weeks)
    - 4~5 different agent or memory systems. Make some simple benchmarking on LongEvalMem datastes 
- Develop more advanced counterfactual methods (remove/rollback/time-shift.) (2 weeks)
- Proof-of-concept: (2 weeks) 
- Hypothesis: existing memory agents will be unreliable with these memory attacking techniques or perturbations. Empirically show the evidence to support the hypothesis. 

- Phase 2: Zirui (Ray) collected a new snyhtetic perosnalization dataset (1M long history)
    - Phase 2, counterfactual analysis on this new dataset (extreme condition)

# Week 2
## Note
- Define non-trival counterfactual operation: remove/rollback/time-shift.
- How we can track this thing in an AI system?
    - Memory influence score: which memories truly drive outputs
    - Dominance rate: Is the agent’s behavior controlled by just one (or a tiny number of) memories?
    - Effective temporal dependency length: How far back in time does memory still affect answers?
- How to pick such memory candidates?


## Progress
- Test MemoryOS on LongMemEval
- Do some basic counterfactual operation to get some results.

# Week 1
## DK's feedback
First, with simple search, I could find many relevant, recent work on long-term memory management [1-6]. Please carefully read them and find differences of yours against them. The contradictory setup seems quite similar to adversarial setup in [1]. Also, [6] seems to have already collected a Conflict Resolution dataset called FactConsolidation. Creating yet another conflict-related benchmark or better agent seems quite redundant to many other prior work.

[1] [2402.17753] Evaluating Very Long-Term Conversational Memory of LLM Agents
[2] Towards Lifelong Dialogue Agents via Timeline-based Memory Management - ACL Anthology
[3] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue - ACL Anthology 
[4] [2503.08026] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents 
[5] [2509.25911] Mem-α: Learning Memory Construction via Reinforcement Learning 
[6] Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 
[7] [2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory 

## Idea 1: Counterfactual Memory Auditing for Long Term Agents
Current long term memory evaluations measure whether an agent can retrieve or update facts, but they do not measure which memories actually influence decisions. As a result, agents may rely heavily on stale, incidental, or low quality memories even when conflict resolution is correct. We propose counterfactual memory auditing, a causal evaluation framework that measures the behavioral impact of individual memory items on agent outputs.
Given an agent with stored memory and a fixed query, we generate counterfactual executions where a specific memory item is removed, rolled back to an earlier version, or temporally shifted. We then measure output divergence across these counterfactuals to compute memory influence scores, dominance rates, and effective temporal dependency lengths. This reveals whether a small subset of memories disproportionately control agent behavior, independent of whether they are contradictory.
We apply this framework to LongMemEval [7] and LoCoMo [1] by auditing preference and profile related memories. Our key hypothesis is that many long term agents exhibit high memory dominance, where a single outdated memory flips downstream behavior, and that common mitigation strategies like append only or overwrite policies do not address this failure mode. This reframes long term memory from a retrieval problem into a causal responsibility problem, introducing a new evaluation axis for memory systems.

