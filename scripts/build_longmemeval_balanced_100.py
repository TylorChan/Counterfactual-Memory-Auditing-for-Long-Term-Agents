#!/usr/bin/env python3
"""Build a reproducible balanced 100-question LongMemEval slice.

The slice keeps the existing balanced 50-question set and adds 50 metadata-only
selected questions from the full cleaned split. Selection is stratified by
question_type and prioritizes questions with stronger long-memory signal
without using any agent outputs.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


QUESTION_TYPE_TARGETS = {
    "temporal-reasoning": 18,
    "multi-session": 18,
    "knowledge-update": 16,
    "single-session-assistant": 16,
    "single-session-preference": 16,
    "single-session-user": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("LongMemEval/data/longmemeval_s_cleaned.json"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("LongMemEval/data/longmemeval_s_cleaned_50.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json"),
    )
    parser.add_argument(
        "--additions-out",
        type=Path,
        default=Path("LongMemEval/data/longmemeval_s_cleaned_extra50_balanced_seed8980.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=8980)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def parse_datetime(value: str) -> dt.datetime | None:
    try:
        normalized = re.sub(r" \([^)]+\)", "", value)
        return dt.datetime.strptime(normalized, "%Y/%m/%d %H:%M")
    except Exception:
        return None


def stable_jitter(question_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0
    lower_or_equal = sum(1 for item in values if item <= value)
    return (lower_or_equal - 1) / (len(values) - 1)


def item_metrics(item: dict[str, Any]) -> dict[str, float]:
    session_ids = item.get("haystack_session_ids") or []
    answer_session_ids = item.get("answer_session_ids") or []
    sid_to_index = {session_id: index for index, session_id in enumerate(session_ids)}
    answer_indices = [
        sid_to_index[session_id]
        for session_id in answer_session_ids
        if session_id in sid_to_index
    ]
    question_time = parse_datetime(str(item.get("question_date", "")))
    evidence_ages = []
    if question_time is not None:
        dates = item.get("haystack_dates") or []
        for index in answer_indices:
            if index >= len(dates):
                continue
            evidence_time = parse_datetime(str(dates[index]))
            if evidence_time is None:
                continue
            evidence_ages.append(max(0.0, (question_time - evidence_time).total_seconds() / 86400.0))

    answer_span = 0
    if answer_indices:
        answer_span = max(answer_indices) - min(answer_indices) + 1

    haystack_sessions = item.get("haystack_sessions") or []
    haystack_chars = 0
    for session in haystack_sessions:
        for message in session:
            if isinstance(message, dict):
                haystack_chars += len(str(message.get("content", "")))

    return {
        "answer_session_count": float(len(answer_session_ids)),
        "mapped_answer_session_count": float(len(answer_indices)),
        "answer_session_span": float(answer_span),
        "max_evidence_age_days": float(max(evidence_ages) if evidence_ages else 0.0),
        "haystack_session_count": float(len(session_ids)),
        "haystack_chars": float(haystack_chars),
        "answer_chars": float(len(str(item.get("answer", "")))),
    }


def weighted_score(
    question_type: str,
    metrics: dict[str, float],
    distributions: dict[str, list[float]],
    question_id: str,
    seed: int,
) -> float:
    ranks = {
        key: percentile_rank(value, distributions[key])
        for key, value in metrics.items()
        if key in distributions
    }
    if question_type == "multi-session":
        score = (
            0.35 * ranks["mapped_answer_session_count"]
            + 0.30 * ranks["answer_session_span"]
            + 0.25 * ranks["max_evidence_age_days"]
            + 0.10 * ranks["haystack_session_count"]
        )
    elif question_type == "temporal-reasoning":
        score = (
            0.40 * ranks["max_evidence_age_days"]
            + 0.25 * ranks["answer_session_span"]
            + 0.25 * ranks["mapped_answer_session_count"]
            + 0.10 * ranks["haystack_session_count"]
        )
    elif question_type == "knowledge-update":
        score = (
            0.45 * ranks["max_evidence_age_days"]
            + 0.35 * ranks["answer_session_span"]
            + 0.10 * ranks["answer_chars"]
            + 0.10 * ranks["haystack_session_count"]
        )
    else:
        score = (
            0.45 * ranks["max_evidence_age_days"]
            + 0.25 * ranks["haystack_session_count"]
            + 0.20 * ranks["answer_chars"]
            + 0.10 * ranks["haystack_chars"]
        )
    return score + 0.01 * stable_jitter(question_id, seed)


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = collections.Counter(item["question_type"] for item in items)
    metrics = [item_metrics(item) for item in items]
    summary: dict[str, Any] = {
        "count": len(items),
        "question_type_counts": dict(sorted(by_type.items())),
    }
    for key in [
        "answer_session_count",
        "answer_session_span",
        "max_evidence_age_days",
        "haystack_session_count",
        "answer_chars",
    ]:
        values = sorted(metric[key] for metric in metrics)
        if values:
            summary[key] = {
                "min": values[0],
                "median": values[len(values) // 2],
                "mean": sum(values) / len(values),
                "max": values[-1],
            }
    return summary


def main() -> None:
    args = parse_args()
    full = load_json(args.full)
    base = load_json(args.base)
    base_ids = {item["question_id"] for item in base}
    full_by_id = {item["question_id"]: item for item in full}
    missing = sorted(base_ids - set(full_by_id))
    if missing:
        raise RuntimeError(f"Base questions missing from full split: {missing[:10]}")

    base_counts = collections.Counter(item["question_type"] for item in base)
    additions_needed = {
        question_type: QUESTION_TYPE_TARGETS[question_type] - base_counts[question_type]
        for question_type in QUESTION_TYPE_TARGETS
    }
    if any(count < 0 for count in additions_needed.values()):
        raise RuntimeError(f"Base already exceeds target quotas: {additions_needed}")

    candidates_by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in full:
        if item["question_id"] in base_ids:
            continue
        candidates_by_type[item["question_type"]].append(item)

    selected_additions: list[dict[str, Any]] = []
    selected_manifest: list[dict[str, Any]] = []
    for question_type, needed in additions_needed.items():
        candidates = candidates_by_type[question_type]
        if len(candidates) < needed:
            raise RuntimeError(
                f"Not enough candidates for {question_type}: need {needed}, have {len(candidates)}"
            )
        candidate_metrics = {item["question_id"]: item_metrics(item) for item in candidates}
        distributions = collections.defaultdict(list)
        for metrics in candidate_metrics.values():
            for key, value in metrics.items():
                if math.isfinite(value):
                    distributions[key].append(value)
        scored = []
        for item in candidates:
            question_id = item["question_id"]
            metrics = candidate_metrics[question_id]
            score = weighted_score(question_type, metrics, distributions, question_id, args.seed)
            scored.append((score, stable_jitter(question_id, args.seed), item, metrics))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]["question_id"]))
        chosen = scored[:needed]
        selected_additions.extend(item for _, _, item, _ in chosen)
        for rank, (score, _, item, metrics) in enumerate(chosen, start=1):
            selected_manifest.append(
                {
                    "question_id": item["question_id"],
                    "question_type": question_type,
                    "rank_within_type": rank,
                    "selection_score": score,
                    "metrics": metrics,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                }
            )

    full_order = {item["question_id"]: index for index, item in enumerate(full)}
    selected_ids = {item["question_id"] for item in selected_additions}
    merged = [item for item in full if item["question_id"] in base_ids or item["question_id"] in selected_ids]
    additions = [item for item in full if item["question_id"] in selected_ids]

    if len(merged) != 100:
        raise RuntimeError(f"Expected 100 merged items, got {len(merged)}")
    if len(additions) != 50:
        raise RuntimeError(f"Expected 50 additions, got {len(additions)}")
    if len({item["question_id"] for item in merged}) != 100:
        raise RuntimeError("Merged output contains duplicate question_id values")
    if any(item["question_id"] not in full_order for item in merged):
        raise RuntimeError("Merged output contains question not in full split")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.additions_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    args.additions_out.write_text(json.dumps(additions, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "selection_name": "longmemeval_s_cleaned_100_balanced_seed8980",
        "seed": args.seed,
        "source_full": str(args.full),
        "source_base": str(args.base),
        "output": str(args.out),
        "additions_output": str(args.additions_out),
        "selection_protocol": [
            "Keep all 50 questions from the existing balanced subset.",
            "Exclude existing question_id values from the full 500-question cleaned split.",
            "Add questions by question_type to double the existing balanced quotas.",
            "Rank candidate questions using only dataset metadata: evidence age, answer-session span, evidence count, history size, and answer length.",
            "Do not use any agent prediction, correctness, or counterfactual result for selection.",
        ],
        "target_question_type_counts": QUESTION_TYPE_TARGETS,
        "base_summary": summarize(base),
        "additions_summary": summarize(additions),
        "merged_summary": summarize(merged),
        "selected_additions": sorted(selected_manifest, key=lambda row: (row["question_type"], row["rank_within_type"])),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(manifest["merged_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
