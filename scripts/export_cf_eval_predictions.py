#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_queries(paths: List[Path]) -> Dict[str, dict]:
    by_qid: Dict[str, dict] = {}
    for path in paths:
        for rec in read_jsonl(path):
            if rec.get("trace_kind") != "cf_query_summary":
                continue
            by_qid[rec["question_id"]] = rec
    return by_qid


def run_sort_key(rec: dict) -> Tuple:
    return (
        float(rec.get("influence_score", 0.0)),
        int(bool(rec.get("answer_changed", False))),
        int(bool(rec.get("dominant", False))),
        int(bool(rec.get("target_was_exposed", False))),
        rec.get("rule_id", ""),
        rec.get("target_write_id", ""),
    )


def load_best_runs(paths: List[Path]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for path in paths:
        for rec in read_jsonl(path):
            if rec.get("trace_kind") != "cf_run":
                continue
            qid = rec["question_id"]
            cur = best.get(qid)
            if cur is None or run_sort_key(rec) > run_sort_key(cur):
                best[qid] = rec
    return best


def main():
    ap = argparse.ArgumentParser(description="Export eval-ready predictions from CF outputs.")
    ap.add_argument("--cf-queries", nargs="+", required=True, help="CF query summary jsonl files")
    ap.add_argument("--cf-runs", nargs="+", required=True, help="CF run jsonl files")
    ap.add_argument("--out", required=True, help="Output prediction jsonl path")
    ap.add_argument("--mode", choices=["max_influence", "baseline_only"], default="max_influence")
    args = ap.parse_args()

    query_paths = [Path(x) for x in args.cf_queries]
    run_paths = [Path(x) for x in args.cf_runs]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(query_paths)
    best_runs = load_best_runs(run_paths) if args.mode == "max_influence" else {}

    with out_path.open("w") as out_f:
        for qid in sorted(queries):
            qrec = queries[qid]
            if args.mode == "max_influence" and qid in best_runs:
                hyp = best_runs[qid].get("cf_answer", qrec.get("baseline_answer", "I don't know."))
            else:
                hyp = qrec.get("baseline_answer", "I don't know.")
            out_f.write(json.dumps({"question_id": qid, "hypothesis": hyp}) + "\n")

    print(out_path)
    print(f"queries={len(queries)} runs={len(best_runs)} mode={args.mode}")


if __name__ == "__main__":
    main()
