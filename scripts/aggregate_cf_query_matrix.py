#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def has_primary_retrieval_schema(row: Dict) -> bool:
    return (
        "baseline_primary_retrieval_write_ids" in row
        or "baseline_retrieval_overlap_gold_write_ids" in row
        or "baseline_exposure_correct" in row
    )


def first_present(obj: Dict, *keys: str):
    for key in keys:
        if key in obj:
            return obj.get(key)
    return None


def aggregate_query_matrix(
    summaries: Iterable[Dict],
    runs: Iterable[Dict],
) -> Dict:
    runs_by_qid: Dict[str, List[Dict]] = {}
    for run in runs:
        qid = str(run.get("question_id") or "").strip()
        if not qid:
            continue
        runs_by_qid.setdefault(qid, []).append(run)

    counts = {
        "retrieved_correct_dominant": 0,
        "retrieved_correct_non_dominant": 0,
        "retrieved_incorrect_dominant": 0,
        "retrieved_incorrect_non_dominant": 0,
    }
    dominance_label_counts = {
        "gold_dominant": 0,
        "non_gold_dominant": 0,
        "ambiguous": 0,
        "no_effect": 0,
    }
    consistency_issue_counts: Dict[str, int] = {}
    query_rows: List[Dict] = []
    primary_schema_count = 0
    exposure_correct_count = 0
    exposure_known_count = 0

    for summary in summaries:
        qid = str(summary.get("question_id") or "").strip()
        if not qid:
            continue
        if has_primary_retrieval_schema(summary):
            primary_schema_count += 1
        exposure_correct_value = first_present(
            summary,
            "baseline_exposure_correct_repaired",
            "baseline_exposure_correct",
        )
        if exposure_correct_value is not None:
            exposure_known_count += 1
            if normalize_bool(exposure_correct_value):
                exposure_correct_count += 1
        retrieval_correct = normalize_bool(
            first_present(
                summary,
                "baseline_retrieval_correct_repaired",
                "baseline_retrieval_correct",
            )
        )
        dominance_label = first_present(
            summary,
            "query_dominance_label_repaired",
            "query_dominance_label",
        )
        gold_dominant = first_present(summary, "query_gold_dominant_repaired", "query_gold_dominant")
        non_gold_dominant = first_present(
            summary,
            "query_non_gold_dominant_repaired",
            "query_non_gold_dominant",
        )
        if dominance_label is None and (gold_dominant is None or non_gold_dominant is None):
            q_runs = runs_by_qid.get(qid, [])
            gold_dominant = any(
                run.get("rule_id") == "rollback_skip"
                and normalize_bool(run.get("dominant"))
                and normalize_bool(
                    first_present(run, "target_is_gold_support_repaired", "target_is_gold_support")
                )
                for run in q_runs
            )
            non_gold_dominant = any(
                run.get("rule_id") == "rollback_skip"
                and normalize_bool(run.get("dominant"))
                and not normalize_bool(
                    first_present(run, "target_is_gold_support_repaired", "target_is_gold_support")
                )
                for run in q_runs
            )
            if gold_dominant:
                dominance_label = "gold_dominant"
            elif non_gold_dominant:
                dominance_label = "non_gold_dominant"
            else:
                dominance_label = "no_effect"
        elif dominance_label is None:
            if normalize_bool(gold_dominant):
                dominance_label = "gold_dominant"
            elif normalize_bool(non_gold_dominant):
                dominance_label = "non_gold_dominant"
            else:
                dominance_label = "no_effect"

        dominance_label = str(dominance_label)
        if dominance_label in dominance_label_counts:
            dominance_label_counts[dominance_label] += 1

        consistency_issues = first_present(summary, "consistency_issues_repaired", "consistency_issues") or []
        for issue in consistency_issues:
            issue_name = str(issue)
            consistency_issue_counts[issue_name] = consistency_issue_counts.get(issue_name, 0) + 1

        if retrieval_correct and dominance_label == "gold_dominant":
            cell = "retrieved_correct_dominant"
        elif retrieval_correct:
            cell = "retrieved_correct_non_dominant"
        elif dominance_label in {"gold_dominant", "non_gold_dominant"}:
            cell = "retrieved_incorrect_dominant"
        else:
            cell = "retrieved_incorrect_non_dominant"
        counts[cell] += 1
        query_rows.append(
            {
                "question_id": qid,
                "baseline_retrieval_correct": retrieval_correct,
                "baseline_exposure_correct": (
                    normalize_bool(exposure_correct_value) if exposure_correct_value is not None else None
                ),
                "query_gold_dominant": dominance_label == "gold_dominant",
                "query_non_gold_dominant": dominance_label == "non_gold_dominant",
                "query_dominance_label": dominance_label,
                "query_confusion_cell": cell,
                "consistency_issues": consistency_issues,
            }
        )

    total = len(query_rows)
    row_totals = {
        "retrieved_correct": counts["retrieved_correct_dominant"] + counts["retrieved_correct_non_dominant"],
        "retrieved_incorrect": counts["retrieved_incorrect_dominant"] + counts["retrieved_incorrect_non_dominant"],
    }
    row_percentages = {
        "retrieved_correct": {
            "dominant": (
                counts["retrieved_correct_dominant"] / row_totals["retrieved_correct"]
                if row_totals["retrieved_correct"]
                else 0.0
            ),
            "non_dominant": (
                counts["retrieved_correct_non_dominant"] / row_totals["retrieved_correct"]
                if row_totals["retrieved_correct"]
                else 0.0
            ),
        },
        "retrieved_incorrect": {
            "dominant": (
                counts["retrieved_incorrect_dominant"] / row_totals["retrieved_incorrect"]
                if row_totals["retrieved_incorrect"]
                else 0.0
            ),
            "non_dominant": (
                counts["retrieved_incorrect_non_dominant"] / row_totals["retrieved_incorrect"]
                if row_totals["retrieved_incorrect"]
                else 0.0
            ),
        },
    }
    return {
        "total_queries": total,
        "counts": counts,
        "dominance_label_counts": dominance_label_counts,
        "consistency_issue_counts": consistency_issue_counts,
        "primary_retrieval_schema_rows": primary_schema_count,
        "legacy_retrieval_schema_rows": total - primary_schema_count,
        "baseline_exposure_correct_count": exposure_correct_count,
        "baseline_exposure_known_count": exposure_known_count,
        "row_totals": row_totals,
        "row_percentages": row_percentages,
        "queries": query_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate professor-style query-level 2x2 from CF traces.")
    parser.add_argument("--cf-queries", type=Path, required=True, help="Path to *.cf_queries.jsonl")
    parser.add_argument("--cf-runs", type=Path, default=None, help="Optional path to *.cf_runs.jsonl")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional summary JSON output path")
    parser.add_argument("--out-queries", type=Path, default=None, help="Optional per-query JSONL output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_jsonl(args.cf_queries)
    runs = load_jsonl(args.cf_runs) if args.cf_runs and args.cf_runs.exists() else []
    result = aggregate_query_matrix(summaries, runs)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_queries:
        args.out_queries.parent.mkdir(parents=True, exist_ok=True)
        with args.out_queries.open("w", encoding="utf-8") as handle:
            for row in result["queries"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "queries"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
