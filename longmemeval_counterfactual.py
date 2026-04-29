from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from longmemeval_audit import append_jsonl, normalize_list

CF_TRACE_VERSION = "cf_audit_v2"


def add_cf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-cf-wrapper",
        action="store_true",
        help="Run true write-time replay counterfactuals over native write ingress units.",
    )
    parser.add_argument(
        "--cf-target-scope",
        choices=("prompt", "retrieved", "candidate"),
        default="prompt",
        help="Which baseline ingress-write universe to intervene on.",
    )
    parser.add_argument(
        "--cf-max-writes",
        type=int,
        default=0,
        help="Optional cap on the number of target writes per query. 0 means no cap.",
    )
    parser.add_argument(
        "--cf-dominance-threshold",
        type=float,
        default=0.75,
        help="Influence threshold used to mark a rollback target as causally dominant.",
    )


def derive_cf_paths(trace_path: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if trace_path is None:
        return None, None
    stem = trace_path.name[:-6] if trace_path.name.endswith(".jsonl") else trace_path.name
    run_path = trace_path.with_name(f"{stem}.cf_runs.jsonl")
    query_path = trace_path.with_name(f"{stem}.cf_queries.jsonl")
    return run_path, query_path


def parse_dt(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def format_dt(dt: Optional[datetime], fallback: Optional[str]) -> Optional[str]:
    if dt is None:
        return fallback
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalize_answer(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def answer_distance(a: str, b: str) -> float:
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if na == nb:
        return 0.0
    return 1.0 - SequenceMatcher(None, na, nb).ratio()


def gini(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if float(v) >= 0.0]
    if not xs:
        return 0.0
    xs.sort()
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(xs))
    return max(0.0, (2.0 * weighted) / (n * total) - (n + 1) / n)


def jaccard_delta(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(normalize_list(a))
    set_b = set(normalize_list(b))
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - (len(set_a & set_b) / len(union))


def _answer_is_idk(text: str) -> bool:
    norm = normalize_answer(text)
    return norm in {
        "i don't know.",
        "i don't know",
        "the information provided is not enough.",
        "the information provided is not enough",
    }


@dataclass(frozen=True)
class CounterfactualSpec:
    cf_type: str
    rule_id: str
    target_write_id: str
    new_timestamp: Optional[str] = None
    note: Optional[str] = None


RULES_BY_QTYPE: Dict[str, Tuple[str, ...]] = {
    "temporal-reasoning": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
        "timeshift_cross_query_boundary",
    ),
    "knowledge-update": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
        "timeshift_cross_query_boundary",
    ),
    "multi-session": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
    ),
    "single-session-assistant": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
    ),
    "single-session-preference": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
    ),
    "single-session-user": (
        "rollback_skip",
        "timeshift_promote_before_query",
        "timeshift_demote_far_past",
    ),
}


# The professor-facing 2x2 should reflect what the agent explicitly fetched,
# not every prompt-side memory item that happened to be exposed downstream.
# These stage/source-form rules define the primary retrieval axis per agent.
PRIMARY_RETRIEVAL_ITEM_RULES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "memoryos": {
        "stages": ("retrieved_page",),
        "source_forms": ("memoryos_page",),
    },
    "theanine": {
        "stages": ("before_refinement",),
        "source_forms": ("theanine_before_refinement",),
    },
    "share": {
        "stages": ("selected_memory",),
        "source_forms": ("share_memory_summary",),
    },
    "ldagent": {
        "stages": ("answer_context", "answer_related"),
        "source_forms": ("context_memory", "related_memory", "ldagent_raw_dialog", "ldagent_summary"),
    },
    "anna": {
        "stages": ("anna_long_term_retrieval",),
        "source_forms": ("anna_retrieved_text",),
    },
    "mem0": {
        "stages": ("mem0_search_result",),
        "source_forms": ("mem0_memory",),
    },
    "mem0_official": {
        "stages": ("mem0_official_search_result",),
        "source_forms": ("mem0_official_memory",),
    },
}


def _scope_ids(query_record: Dict, scope: str) -> List[str]:
    prompt_ids = normalize_list(query_record.get("prompt_write_ids"))
    retrieved_ids = normalize_list(query_record.get("retrieved_write_ids"))
    candidate_ids = normalize_list(query_record.get("candidate_write_ids"))
    if scope == "candidate":
        return candidate_ids
    if scope == "retrieved":
        return retrieved_ids or prompt_ids or candidate_ids
    return prompt_ids or retrieved_ids or candidate_ids


def _is_primary_retrieval_item(agent: str, item: Dict) -> bool:
    rules = PRIMARY_RETRIEVAL_ITEM_RULES.get(agent, {})
    stages = set(rules.get("stages") or ())
    source_forms = set(rules.get("source_forms") or ())
    stage = str(item.get("stage") or "").strip()
    source_form = str(item.get("source_form") or "").strip()
    if stages or source_forms:
        return (stage in stages) or (source_form in source_forms)
    return bool(item.get("audit_eligible")) and bool(normalize_list(item.get("source_write_ids")))


def _primary_retrieval_write_ids(agent: str, query_record: Dict) -> List[str]:
    write_ids: List[str] = []
    for item in (query_record.get("retrieved_items") or []):
        if not _is_primary_retrieval_item(agent, item):
            continue
        write_ids.extend(normalize_list(item.get("source_write_ids")))
    return normalize_list(write_ids)


def _match_answer_sessions(write_record: Dict, answer_session_ids: Sequence[str]) -> bool:
    answer_set = set(normalize_list(answer_session_ids))
    if not answer_set:
        return False
    lineage = set(normalize_list(write_record.get("lineage_source_ids")))
    if lineage & answer_set:
        return True
    session_id = str(write_record.get("session_id") or "").strip()
    if session_id and session_id in answer_set:
        return True
    return False


def build_cf_specs(
    *,
    question_type: str,
    query_record: Dict,
    write_records: Sequence[Dict],
    answer_session_ids: Sequence[str],
    max_writes: int,
    scope: str,
) -> List[CounterfactualSpec]:
    writes_by_id = {record["write_id"]: record for record in write_records}
    target_ids = [wid for wid in _scope_ids(query_record, scope) if wid in writes_by_id]
    if target_ids:
        gold_target_ids = [
            wid for wid in target_ids if _match_answer_sessions(writes_by_id[wid], answer_session_ids)
        ]
        non_gold_target_ids = [wid for wid in target_ids if wid not in set(gold_target_ids)]
        target_ids = gold_target_ids + non_gold_target_ids
    if max_writes > 0:
        target_ids = target_ids[:max_writes]

    query_dt = parse_dt(query_record.get("query_time") or query_record.get("question_date_used"))
    timestamps = [parse_dt(writes_by_id[wid].get("timestamp")) for wid in target_ids]
    timestamps = [dt for dt in timestamps if dt is not None]
    oldest_dt = min(timestamps) if timestamps else query_dt

    specs: List[CounterfactualSpec] = []
    rules = RULES_BY_QTYPE.get(
        question_type,
        ("rollback_skip", "timeshift_promote_before_query", "timeshift_demote_far_past"),
    )

    for wid in target_ids:
        record = writes_by_id[wid]
        ts = parse_dt(record.get("timestamp"))
        is_gold = _match_answer_sessions(record, answer_session_ids)
        for rule in rules:
            if rule == "rollback_skip":
                specs.append(CounterfactualSpec(cf_type="rollback", rule_id=rule, target_write_id=wid))
                continue
            if ts is None or query_dt is None:
                continue
            if rule == "timeshift_promote_before_query":
                new_dt = query_dt - timedelta(seconds=1)
            elif rule == "timeshift_demote_far_past":
                anchor = oldest_dt or ts
                new_dt = anchor - timedelta(days=30)
            elif rule == "timeshift_cross_query_boundary":
                if question_type not in {"temporal-reasoning", "knowledge-update"}:
                    continue
                new_dt = query_dt + timedelta(seconds=1)
            else:
                continue
            specs.append(
                CounterfactualSpec(
                    cf_type="time_shift",
                    rule_id=rule,
                    target_write_id=wid,
                    new_timestamp=format_dt(new_dt, None),
                    note="gold_support" if is_gold else "non_gold_support",
                )
            )
    return specs


def compute_influence(
    *,
    baseline_answer: str,
    cf_answer: str,
    baseline_retrieved_ids: Sequence[str],
    cf_retrieved_ids: Sequence[str],
    baseline_prompt_ids: Sequence[str],
    cf_prompt_ids: Sequence[str],
) -> Dict:
    answer_changed = int(normalize_answer(baseline_answer) != normalize_answer(cf_answer))
    abstention_flip = int(_answer_is_idk(baseline_answer) != _answer_is_idk(cf_answer))
    retrieved_delta = jaccard_delta(baseline_retrieved_ids, cf_retrieved_ids)
    prompt_delta = jaccard_delta(baseline_prompt_ids, cf_prompt_ids)
    distance = answer_distance(baseline_answer, cf_answer)
    score = (
        1.0 * answer_changed
        + 0.5 * abstention_flip
        + 0.5 * distance
        + 0.25 * retrieved_delta
        + 0.25 * prompt_delta
    )
    return {
        "answer_changed": answer_changed,
        "abstention_flip": abstention_flip,
        "retrieved_write_set_change": int(retrieved_delta > 0.0),
        "prompt_write_set_change": int(prompt_delta > 0.0),
        "retrieved_write_jaccard_delta": retrieved_delta,
        "prompt_write_jaccard_delta": prompt_delta,
        "answer_distance": distance,
        "influence_score": score,
    }


def classify_query_dominance(
    rollback_runs: Sequence[Dict],
    minimum_effect: float,
    *,
    top_share_threshold: float = 0.55,
    margin_threshold: float = 0.15,
) -> Dict:
    ordered = sorted(
        (
            run
            for run in rollback_runs
            if run.get("rule_id") == "rollback_skip"
        ),
        key=lambda run: (
            float(run.get("influence_score", 0.0)),
            int(bool(run.get("answer_changed", False))),
            int(bool(run.get("target_is_gold_support", False))),
            str(run.get("target_write_id") or ""),
        ),
        reverse=True,
    )
    if not ordered:
        return {
            "label": "no_effect",
            "dominant_write_ids": [],
            "gold_dominant_write_ids": [],
            "non_gold_dominant_write_ids": [],
            "top1_score": 0.0,
            "top2_score": 0.0,
            "top_share": 0.0,
            "margin": 0.0,
        }

    scores = [float(run.get("influence_score", 0.0)) for run in ordered]
    total = sum(scores)
    top1_score = scores[0]
    top2_score = scores[1] if len(scores) > 1 else 0.0
    margin = top1_score - top2_score
    top_share = (top1_score / total) if total > 0 else 0.0

    if top1_score <= minimum_effect:
        label = "no_effect"
        dominant_runs: List[Dict] = []
    elif top_share < top_share_threshold and margin < margin_threshold:
        label = "ambiguous"
        dominant_runs = []
    else:
        dominant_runs = [ordered[0]]
        label = "gold_dominant" if ordered[0].get("target_is_gold_support") else "non_gold_dominant"

    dominant_write_ids = [str(run.get("target_write_id")) for run in dominant_runs if run.get("target_write_id")]
    gold_dominant_write_ids = [
        str(run.get("target_write_id"))
        for run in dominant_runs
        if run.get("target_is_gold_support") and run.get("target_write_id")
    ]
    non_gold_dominant_write_ids = [
        str(run.get("target_write_id"))
        for run in dominant_runs
        if not run.get("target_is_gold_support") and run.get("target_write_id")
    ]
    return {
        "label": label,
        "dominant_write_ids": dominant_write_ids,
        "gold_dominant_write_ids": gold_dominant_write_ids,
        "non_gold_dominant_write_ids": non_gold_dominant_write_ids,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top_share": top_share,
        "margin": margin,
    }


def summarize_replay_cf(
    *,
    agent: str,
    entry: Dict,
    baseline_query_record: Dict,
    write_records: Sequence[Dict],
    cf_results: Sequence[Dict],
    dominance_threshold: float,
) -> Tuple[List[Dict], Dict]:
    writes_by_id = {record["write_id"]: record for record in write_records}
    baseline_answer = str(baseline_query_record.get("baseline_answer") or "")
    baseline_retrieved_ids = normalize_list(baseline_query_record.get("retrieved_write_ids"))
    baseline_prompt_ids = normalize_list(baseline_query_record.get("prompt_write_ids"))
    baseline_exposed_ids = normalize_list([*baseline_retrieved_ids, *baseline_prompt_ids])
    baseline_primary_retrieved_ids = _primary_retrieval_write_ids(agent, baseline_query_record)
    question_type = str(baseline_query_record.get("question_type") or entry.get("question_type") or "unknown")
    query_time = (
        baseline_query_record.get("query_time")
        or baseline_query_record.get("question_date_used")
        or baseline_query_record.get("question_date")
        or entry.get("question_date")
    )
    query_dt = parse_dt(query_time)
    answer_session_ids = normalize_list(entry.get("answer_session_ids"))

    run_records: List[Dict] = []
    rollback_scores: List[float] = []
    rollback_answer_flip_count = 0
    rollback_abstention_flip_count = 0
    rollback_answer_distances: List[float] = []

    for result in cf_results:
        spec: CounterfactualSpec = result["spec"]
        cf_answer = result["cf_answer"]
        cf_retrieved_ids = normalize_list(result.get("cf_retrieved_write_ids"))
        cf_prompt_ids = normalize_list(result.get("cf_prompt_write_ids"))
        target_record = writes_by_id.get(spec.target_write_id, {})
        target_timestamp = result.get("target_timestamp") or spec.new_timestamp or target_record.get("timestamp")
        target_dt = parse_dt(target_timestamp)
        age_seconds = None
        if query_dt is not None and target_dt is not None:
            delta = (query_dt - target_dt).total_seconds()
            if delta >= 0:
                age_seconds = delta

        is_gold_support = _match_answer_sessions(target_record, answer_session_ids)
        was_retrieved = spec.target_write_id in baseline_retrieved_ids
        was_prompted = spec.target_write_id in baseline_prompt_ids
        was_exposed = spec.target_write_id in baseline_exposed_ids
        influence = compute_influence(
            baseline_answer=baseline_answer,
            cf_answer=cf_answer,
            baseline_retrieved_ids=baseline_retrieved_ids,
            cf_retrieved_ids=cf_retrieved_ids,
            baseline_prompt_ids=baseline_prompt_ids,
            cf_prompt_ids=cf_prompt_ids,
        )
        if spec.rule_id == "rollback_skip":
            rollback_scores.append(influence["influence_score"])
            rollback_answer_flip_count += influence["answer_changed"]
            rollback_abstention_flip_count += influence["abstention_flip"]
            rollback_answer_distances.append(influence["answer_distance"])

        run_records.append(
            {
                "trace_version": CF_TRACE_VERSION,
                "trace_kind": "cf_run",
                "agent": agent,
                "question_id": entry.get("question_id"),
                "question_type": question_type,
                "query_time": query_time,
                "baseline_answer": baseline_answer,
                "cf_answer": cf_answer,
                "cf_type": spec.cf_type,
                "rule_id": spec.rule_id,
                "target_write_id": spec.target_write_id,
                "target_write_type": target_record.get("write_type"),
                "target_timestamp": target_timestamp,
                "target_write_timestamp": target_timestamp,
                "target_is_gold_support": is_gold_support,
                "target_was_retrieved": was_retrieved,
                "target_was_prompted": was_prompted,
                "target_was_exposed": was_exposed,
                "baseline_retrieved_write_ids": baseline_retrieved_ids,
                "baseline_prompt_write_ids": baseline_prompt_ids,
                "cf_retrieved_write_ids": cf_retrieved_ids,
                "cf_prompt_write_ids": cf_prompt_ids,
                **influence,
                "dominant": False,
                "age_seconds": age_seconds,
                "cf_extra": result.get("cf_extra") or {},
            }
        )

    rollback_run_records = [record for record in run_records if record["rule_id"] == "rollback_skip"]
    dominance = classify_query_dominance(rollback_run_records, dominance_threshold)
    dominant_write_id_set = set(dominance["dominant_write_ids"])
    dominant_ages_seconds: List[float] = []
    gold_support_write_ids = [
        record["write_id"]
        for record in write_records
        if _match_answer_sessions(record, answer_session_ids)
    ]
    gold_support_write_id_set = set(gold_support_write_ids)
    confusion = {
        "retrieved_correct_dominant": 0,
        "retrieved_correct_non_dominant": 0,
        "retrieved_incorrect_dominant": 0,
        "retrieved_incorrect_non_dominant": 0,
    }

    for record in run_records:
        if record["rule_id"] != "rollback_skip":
            continue
        record["dominant"] = record["target_write_id"] in dominant_write_id_set
        if record["dominant"] and record.get("age_seconds") is not None:
            dominant_ages_seconds.append(float(record["age_seconds"]))

    baseline_exposure_correct = bool(set(baseline_exposed_ids) & gold_support_write_id_set)
    baseline_retrieval_correct = bool(set(baseline_primary_retrieved_ids) & gold_support_write_id_set)
    consistency_issues: List[str] = []
    if dominance["label"] == "gold_dominant" and not baseline_retrieval_correct:
        consistency_issues.append("gold_dominant_without_gold_retrieval")
    if baseline_retrieval_correct and dominance["label"] == "gold_dominant":
        confusion["retrieved_correct_dominant"] = 1
    elif baseline_retrieval_correct:
        confusion["retrieved_correct_non_dominant"] = 1
    elif dominance["label"] in {"gold_dominant", "non_gold_dominant"}:
        confusion["retrieved_incorrect_dominant"] = 1
    else:
        confusion["retrieved_incorrect_non_dominant"] = 1

    summary = {
        "trace_version": CF_TRACE_VERSION,
        "trace_kind": "cf_query_summary",
        "agent": agent,
        "question_id": entry.get("question_id"),
        "question_type": question_type,
        "query_time": query_time,
        "question_date": query_time,
        "baseline_query_timestamp": query_time,
        "baseline_answer": baseline_answer,
        "baseline_retrieval_correct": baseline_retrieval_correct,
        "baseline_exposure_correct": baseline_exposure_correct,
        "baseline_primary_retrieval_write_ids": baseline_primary_retrieved_ids,
        "baseline_primary_retrieval_item_count": sum(
            1 for item in (baseline_query_record.get("retrieved_items") or []) if _is_primary_retrieval_item(agent, item)
        ),
        "baseline_retrieval_overlap_gold_write_ids": normalize_list(
            set(baseline_primary_retrieved_ids) & gold_support_write_id_set
        ),
        "baseline_exposure_overlap_gold_write_ids": normalize_list(
            set(baseline_exposed_ids) & gold_support_write_id_set
        ),
        "gold_support_write_ids": gold_support_write_ids,
        "rollback_gini": gini(rollback_scores),
        "rollback_influence_scores": rollback_scores,
        "rollback_mean_influence": (sum(rollback_scores) / len(rollback_scores)) if rollback_scores else 0.0,
        "rollback_max_influence": max(rollback_scores) if rollback_scores else 0.0,
        "rollback_answer_flip_count": rollback_answer_flip_count,
        "rollback_answer_flip_rate": (
            rollback_answer_flip_count / len(rollback_scores) if rollback_scores else 0.0
        ),
        "rollback_abstention_flip_count": rollback_abstention_flip_count,
        "rollback_abstention_flip_rate": (
            rollback_abstention_flip_count / len(rollback_scores) if rollback_scores else 0.0
        ),
        "rollback_mean_answer_distance": (
            sum(rollback_answer_distances) / len(rollback_answer_distances)
            if rollback_answer_distances
            else 0.0
        ),
        "rollback_max_answer_distance": max(rollback_answer_distances) if rollback_answer_distances else 0.0,
        "query_fragile": rollback_answer_flip_count > 0,
        "dominant_write_ids": dominance["dominant_write_ids"],
        "gold_dominant_write_ids": dominance["gold_dominant_write_ids"],
        "non_gold_dominant_write_ids": dominance["non_gold_dominant_write_ids"],
        "query_dominance_label": dominance["label"],
        "query_gold_dominant": dominance["label"] == "gold_dominant",
        "query_non_gold_dominant": dominance["label"] == "non_gold_dominant",
        "query_ambiguous_dominance": dominance["label"] == "ambiguous",
        "query_no_effect": dominance["label"] == "no_effect",
        "dominance_top1_score": dominance["top1_score"],
        "dominance_top2_score": dominance["top2_score"],
        "dominance_top_share": dominance["top_share"],
        "dominance_margin": dominance["margin"],
        "etdl_seconds": max(dominant_ages_seconds) if dominant_ages_seconds else None,
        "confusion_matrix": confusion,
        "consistency_issues": consistency_issues,
        "cf_run_count": len(run_records),
        "retrieved_item_count": len(baseline_query_record.get("retrieved_items") or []),
        "prompt_item_count": len(baseline_query_record.get("prompt_items") or []),
        "retrieved_item_coverage": (
            sum(
                1
                for item in (baseline_query_record.get("retrieved_items") or [])
                if item.get("audit_eligible") and normalize_list(item.get("source_write_ids"))
            )
            / len(baseline_query_record.get("retrieved_items") or [])
            if (baseline_query_record.get("retrieved_items") or [])
            else 0.0
        ),
        "prompt_item_coverage": (
            sum(
                1
                for item in (baseline_query_record.get("prompt_items") or [])
                if item.get("audit_eligible") and normalize_list(item.get("source_write_ids"))
            )
            / len(baseline_query_record.get("prompt_items") or [])
            if (baseline_query_record.get("prompt_items") or [])
            else 0.0
        ),
    }
    summary["query_confusion_cell"] = (
        "retrieved_correct_dominant"
        if summary["baseline_retrieval_correct"] and summary["query_gold_dominant"]
        else "retrieved_correct_non_dominant"
        if summary["baseline_retrieval_correct"]
        else "retrieved_incorrect_dominant"
        if summary["query_gold_dominant"] or summary["query_non_gold_dominant"]
        else "retrieved_incorrect_non_dominant"
    )
    return run_records, summary


def append_cf_outputs(
    *,
    run_path: Optional[Path],
    query_path: Optional[Path],
    run_records: Sequence[Dict],
    query_summary: Dict,
) -> None:
    for record in run_records:
        append_jsonl(run_path, record)
    append_jsonl(query_path, query_summary)
