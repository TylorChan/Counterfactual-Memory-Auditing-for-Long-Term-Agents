from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def survival_points(values: List[float]) -> List[Dict]:
    if not values:
        return []
    ordered = sorted(float(value) for value in values)
    total = len(ordered)
    thresholds = sorted(set(ordered))
    return [
        {
            "threshold_seconds": threshold,
            "survival": sum(1 for value in ordered if value >= threshold) / total,
        }
        for threshold in thresholds
    ]


def first_present(row: Dict, *keys: str):
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def normalize_bool(value) -> bool:
    return bool(value)


def dominance_label(row: Dict) -> str:
    label = first_present(row, "query_dominance_label_repaired", "query_dominance_label")
    if label is not None:
        return str(label)
    if normalize_bool(first_present(row, "query_gold_dominant_repaired", "query_gold_dominant")):
        return "gold_dominant"
    if normalize_bool(first_present(row, "query_non_gold_dominant_repaired", "query_non_gold_dominant")):
        return "non_gold_dominant"
    if normalize_bool(first_present(row, "query_ambiguous_dominance_repaired", "query_ambiguous_dominance")):
        return "ambiguous"
    return "no_effect"


def query_confusion_cell(row: Dict) -> str:
    retrieval_correct = normalize_bool(
        first_present(row, "baseline_retrieval_correct_repaired", "baseline_retrieval_correct")
    )
    label = dominance_label(row)
    if retrieval_correct and label == "gold_dominant":
        return "retrieved_correct_dominant"
    if retrieval_correct:
        return "retrieved_correct_non_dominant"
    if label in {"gold_dominant", "non_gold_dominant"}:
        return "retrieved_incorrect_dominant"
    return "retrieved_incorrect_non_dominant"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cf_query_jsonl", nargs="+")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows: List[Dict] = []
    for raw in args.cf_query_jsonl:
        rows.extend(load_jsonl(Path(raw)))

    by_agent: Dict[str, List[Dict]] = {}
    for row in rows:
        by_agent.setdefault(row.get("agent", "unknown"), []).append(row)

    out: Dict[str, Dict] = {}
    for agent, agent_rows in by_agent.items():
        ginis = [float(row.get("rollback_gini") or 0.0) for row in agent_rows]
        answer_flip_rates = [float(row.get("rollback_answer_flip_rate") or 0.0) for row in agent_rows]
        abstention_flip_rates = [float(row.get("rollback_abstention_flip_rate") or 0.0) for row in agent_rows]
        answer_distances = [float(row.get("rollback_mean_answer_distance") or 0.0) for row in agent_rows]
        fragility_flags = [1.0 if row.get("query_fragile") else 0.0 for row in agent_rows]
        retrieved_coverages = [float(row.get("retrieved_item_coverage") or 0.0) for row in agent_rows]
        prompt_coverages = [float(row.get("prompt_item_coverage") or 0.0) for row in agent_rows]
        etdls = [float(row["etdl_seconds"]) for row in agent_rows if row.get("etdl_seconds") is not None]
        confusion = {
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
        for row in agent_rows:
            cell = query_confusion_cell(row)
            confusion[cell] = confusion.get(cell, 0) + 1
            label = dominance_label(row)
            if label in dominance_label_counts:
                dominance_label_counts[label] += 1
            issues = first_present(row, "consistency_issues_repaired", "consistency_issues") or []
            for issue in issues:
                issue_name = str(issue)
                consistency_issue_counts[issue_name] = consistency_issue_counts.get(issue_name, 0) + 1
        out[agent] = {
            "n_queries": len(agent_rows),
            "answer_flip_rate_mean": mean(answer_flip_rates) if answer_flip_rates else 0.0,
            "query_fragility_rate": mean(fragility_flags) if fragility_flags else 0.0,
            "abstention_flip_rate_mean": mean(abstention_flip_rates) if abstention_flip_rates else 0.0,
            "mean_answer_distance": mean(answer_distances) if answer_distances else 0.0,
            "rollback_gini_mean": mean(ginis) if ginis else 0.0,
            "rollback_gini_median": median(ginis) if ginis else 0.0,
            "retrieved_item_coverage_mean": mean(retrieved_coverages) if retrieved_coverages else 0.0,
            "prompt_item_coverage_mean": mean(prompt_coverages) if prompt_coverages else 0.0,
            "etdl_count": len(etdls),
            "etdl_mean_seconds": mean(etdls) if etdls else None,
            "etdl_median_seconds": median(etdls) if etdls else None,
            "etdl_max_seconds": max(etdls) if etdls else None,
            "etdl_seconds": etdls,
            "etdl_survival_curve": survival_points(etdls),
            "confusion_matrix": confusion,
            "dominance_label_counts": dominance_label_counts,
            "consistency_issue_counts": consistency_issue_counts,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
