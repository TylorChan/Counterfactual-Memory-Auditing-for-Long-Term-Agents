from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

ROOT = Path("/Users/daqingchen/csci8980")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from longmemeval_audit import normalize_list
from memos_longmemeval_bridge.run_infer import iter_qa_pairs, normalize_timestamp


DATASET_PATH = ROOT / "LongMemEval/data/longmemeval_s_cleaned_50.json"
BASELINE_DIR = ROOT / "cf_compare_results/memoryos_original_trace"
CF_DIR = ROOT / "cf_compare_results/memoryos_cf_trace"
OUT_DIR = ROOT / "cf_compare_results/memoryos_cf_repaired"


@dataclass
class QuerySupport:
    gold_support_write_ids: List[str]
    baseline_retrieval_correct: bool
    write_level_confusion: Dict[str, int]


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset() -> Dict[str, Dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return {entry["question_id"]: entry for entry in data}


def build_session_maps(entry: Dict) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    qa_map: Dict[str, Set[str]] = defaultdict(set)
    ts_map: Dict[str, Set[str]] = defaultdict(set)
    for session_id, session_date, turns in zip(
        entry.get("haystack_session_ids", []),
        entry.get("haystack_dates", []),
        entry.get("haystack_sessions", []),
    ):
        normalized_ts = normalize_timestamp(session_date)
        ts_map[normalized_ts].add(session_id)
        for user_input, agent_response in iter_qa_pairs(turns):
            qa_text = f"User: {user_input}\nAssistant: {agent_response}"
            qa_map[qa_text].add(session_id)
    return qa_map, ts_map


def infer_write_sessions(write_record: Dict, qa_map: Dict[str, Set[str]], ts_map: Dict[str, Set[str]]) -> Set[str]:
    matched: Set[str] = set()
    text = (write_record.get("content_text") or "").strip()
    if text in qa_map:
        matched |= qa_map[text]

    timestamp = str(write_record.get("timestamp") or "").strip()
    timestamp_matches = ts_map.get(timestamp, set())
    if len(timestamp_matches) == 1:
        matched |= timestamp_matches

    return matched


def latest_files(pattern: str) -> List[Path]:
    grouped: Dict[str, Path] = {}
    for path in sorted(CF_DIR.glob(pattern)):
        key = path.name.split(".trace.", 1)[0]
        grouped[key] = path
    return sorted(grouped.values())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    baseline_writes: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(BASELINE_DIR.glob("*.audit_writes.jsonl")):
        for row in load_jsonl(path):
            baseline_writes[row["question_id"]].append(row)

    baseline_queries: Dict[str, Dict] = {}
    for path in sorted(BASELINE_DIR.glob("*.audit_queries.jsonl")):
        for row in load_jsonl(path):
            baseline_queries[row["question_id"]] = row

    write_id_to_sessions: Dict[str, Set[str]] = {}
    write_id_to_record: Dict[str, Dict] = {}
    mapping_stats = Counter()

    for question_id, rows in baseline_writes.items():
        qa_map, ts_map = build_session_maps(dataset[question_id])
        for row in rows:
            matched_sessions = infer_write_sessions(row, qa_map, ts_map)
            write_id_to_sessions[row["write_id"]] = matched_sessions
            write_id_to_record[row["write_id"]] = row
            mapping_stats["total_writes"] += 1
            mapping_stats[f"type::{row['write_type']}"] += 1
            if matched_sessions:
                mapping_stats["mapped_writes"] += 1
                mapping_stats[f"mapped::{row['write_type']}"] += 1

    repaired_queries: Dict[str, QuerySupport] = {}
    query_level_stats = Counter()

    for question_id, query_record in baseline_queries.items():
        answer_session_ids = set(normalize_list(dataset[question_id].get("answer_session_ids")))
        prompt_write_ids = normalize_list(query_record.get("prompt_write_ids"))
        gold_support_write_ids = [
            write_id
            for write_id in prompt_write_ids
            if write_id_to_sessions.get(write_id, set()) & answer_session_ids
        ]
        baseline_retrieval_correct = bool(gold_support_write_ids)
        repaired_queries[question_id] = QuerySupport(
            gold_support_write_ids=gold_support_write_ids,
            baseline_retrieval_correct=baseline_retrieval_correct,
            write_level_confusion={
                "retrieved_correct_dominant": 0,
                "retrieved_correct_non_dominant": 0,
                "retrieved_incorrect_dominant": 0,
                "retrieved_incorrect_non_dominant": 0,
            },
        )
        query_level_stats["total_queries"] += 1
        if baseline_retrieval_correct:
            query_level_stats["retrieval_correct_queries"] += 1

    cf_runs_by_question: Dict[str, List[Dict]] = defaultdict(list)
    for path in latest_files("*.cf_runs.jsonl"):
        for row in load_jsonl(path):
            if row.get("rule_id") == "rollback_skip":
                cf_runs_by_question[row["question_id"]].append(row)

    write_level_stats = Counter()
    repaired_run_rows: List[Dict] = []

    for question_id, runs in cf_runs_by_question.items():
        answer_session_ids = set(normalize_list(dataset[question_id].get("answer_session_ids")))
        support = repaired_queries[question_id]
        for row in runs:
            target_write_id = row["target_write_id"]
            target_is_gold = bool(write_id_to_sessions.get(target_write_id, set()) & answer_session_ids)
            dominant = bool(row.get("dominant"))
            if target_is_gold and dominant:
                cell = "retrieved_correct_dominant"
            elif target_is_gold and not dominant:
                cell = "retrieved_correct_non_dominant"
            elif dominant:
                cell = "retrieved_incorrect_dominant"
            else:
                cell = "retrieved_incorrect_non_dominant"

            support.write_level_confusion[cell] += 1
            write_level_stats[cell] += 1

            repaired = dict(row)
            repaired["target_is_gold_support_repaired"] = target_is_gold
            repaired["target_matched_answer_sessions"] = sorted(
                write_id_to_sessions.get(target_write_id, set()) & answer_session_ids
            )
            repaired_run_rows.append(repaired)

    repaired_query_rows: List[Dict] = []
    for question_id, support in repaired_queries.items():
        repaired_query_rows.append(
            {
                "question_id": question_id,
                "gold_support_write_ids_repaired": support.gold_support_write_ids,
                "baseline_retrieval_correct_repaired": support.baseline_retrieval_correct,
                "confusion_matrix_repaired": support.write_level_confusion,
            }
        )

    aggregate = {
        "mapping_stats": mapping_stats,
        "query_level_stats": query_level_stats,
        "write_level_stats": write_level_stats,
    }

    (OUT_DIR / "memoryos_repaired_cf_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUT_DIR / "memoryos_repaired_cf_queries.jsonl").open("w", encoding="utf-8") as f:
        for row in repaired_query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "memoryos_repaired_cf_runs.jsonl").open("w", encoding="utf-8") as f:
        for row in repaired_run_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
