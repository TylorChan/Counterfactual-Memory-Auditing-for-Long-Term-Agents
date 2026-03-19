from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        etdls = [float(row["etdl_seconds"]) for row in agent_rows if row.get("etdl_seconds") is not None]
        confusion = {
            "retrieved_correct_dominant": 0,
            "retrieved_correct_non_dominant": 0,
            "retrieved_incorrect_dominant": 0,
            "retrieved_incorrect_non_dominant": 0,
        }
        for row in agent_rows:
            for key, value in (row.get("confusion_matrix") or {}).items():
                confusion[key] = confusion.get(key, 0) + int(value)
        out[agent] = {
            "n_queries": len(agent_rows),
            "rollback_gini_mean": mean(ginis) if ginis else 0.0,
            "rollback_gini_median": median(ginis) if ginis else 0.0,
            "etdl_seconds": etdls,
            "confusion_matrix": confusion,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
