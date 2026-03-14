#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longmemeval_unified_answer import EvidenceRow, build_unified_qa_messages

SESSION_NAMES = ["first", "second", "third", "fourth", "fifth"]


def session_field_prefix(session_num: int) -> str:
    if 1 <= session_num <= len(SESSION_NAMES):
        return f"{SESSION_NAMES[session_num-1]}_session"
    return f"session_{session_num}"


def load_env_file(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run THEANINE on LongMemEval and export LongMemEval-format predictions."
    )
    parser.add_argument("--theanine-dir", type=Path, required=True)
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("./theanine_longmemeval_bridge/runtime"),
        help="Working directory for generated THEANINE episode/result files.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--history-sessions",
        type=int,
        default=0,
        help="Number of history sessions to replay. Use 0 to replay all haystack sessions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--omit-question-date", action="store_true")
    parser.add_argument("--preserve-session-order", action="store_true")
    parser.add_argument(
        "--verbose-upstream",
        action="store_true",
        help="Show THEANINE internal prints.",
    )
    return parser.parse_args()


@contextmanager
def maybe_silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        yield


def ensure_openai_config(theanine_dir: Path, api_key: str) -> Path:
    conf_dir = theanine_dir / "conf.d"
    conf_dir.mkdir(parents=True, exist_ok=True)
    config_path = conf_dir / "config.yaml"
    config_path.write_text(f"openai:\n  key: {api_key}\n", encoding="utf-8")
    return config_path


def import_theanine_modules(theanine_dir: Path):
    if str(theanine_dir) not in sys.path:
        sys.path.insert(0, str(theanine_dir))
    from src.summarize import Summarizer  # type: ignore
    from src.memory_constructor import MemoryConstructor  # type: ignore
    from src.theanine import Theanine  # type: ignore

    return Summarizer, MemoryConstructor, Theanine


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset from {path}")
    return data


def sorted_session_indices(entry: Dict, preserve_session_order: bool) -> List[int]:
    indices = list(range(len(entry["haystack_sessions"])))
    if preserve_session_order:
        return indices
    return sorted(indices, key=lambda i: entry["haystack_dates"][i])


def to_theanine_speakers(turns: Sequence[Dict]) -> Tuple[List[str], List[str]]:
    dialogue: List[str] = []
    speakers: List[str] = []
    for turn in turns:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            speaker = "Speaker A"
        elif role == "assistant":
            speaker = "Speaker B"
        else:
            continue
        dialogue.append(content)
        speakers.append(speaker)
    return dialogue, speakers


def build_episode(entry: Dict, history_indices: List[int], omit_question_date: bool) -> Dict:
    history_session_count = len(history_indices)
    total_session_count = history_session_count + 1
    episode: Dict[str, object] = {
        "dataID": entry["question_id"],
        "history_session_count": history_session_count,
        "total_session_count": total_session_count,
    }
    for session_num, idx in enumerate(history_indices, start=1):
        prefix = session_field_prefix(session_num)
        dialogue, speakers = to_theanine_speakers(entry["haystack_sessions"][idx])
        episode[f"{prefix}_dialogue"] = dialogue
        episode[f"{prefix}_speakers"] = speakers

    query = entry["question"].strip()
    if not omit_question_date:
        query = f"Current date: {entry['question_date']}\n\n{query}"
    qa_prefix = session_field_prefix(total_session_count)
    episode[f"{qa_prefix}_dialogue"] = [query, ""]
    episode[f"{qa_prefix}_speakers"] = ["Speaker A", "Speaker B"]
    return episode


def build_trace_stub(
    entry: Dict,
    history_sessions: int,
    preserve_session_order: bool,
    omit_question_date: bool,
    seed: int,
) -> Dict:
    ordered_indices = sorted_session_indices(entry, preserve_session_order=preserve_session_order)
    if history_sessions > 0:
        selected_indices = ordered_indices[:history_sessions]
    else:
        selected_indices = ordered_indices

    selected_session_ids = [entry["haystack_session_ids"][i] for i in selected_indices]
    selected_session_dates = [entry["haystack_dates"][i] for i in selected_indices]
    omitted_answer_session_ids = [
        sid for sid in entry.get("answer_session_ids", []) if sid not in set(selected_session_ids)
    ]
    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "history_sessions_requested": history_sessions,
        "history_sessions_used": len(selected_indices),
        "selected_history_indices": selected_indices,
        "selected_session_ids": selected_session_ids,
        "selected_session_dates": selected_session_dates,
        "qa_session_num": len(selected_indices) + 1,
        "answer_session_ids": entry.get("answer_session_ids", []),
        "omitted_answer_session_ids": omitted_answer_session_ids,
        "question_date_used": None if omit_question_date else entry.get("question_date"),
        "seed": seed,
    }


def write_episode_json(path: Path, episode: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([episode], ensure_ascii=False, indent=2), encoding="utf-8")


def run_theanine_for_entry(
    entry: Dict,
    theanine_dir: Path,
    runtime_dir: Path,
    llm_model: str,
    temperature: float,
    history_sessions: int,
    preserve_session_order: bool,
    omit_question_date: bool,
    dry_run: bool,
    verbose_upstream: bool,
    seed: int,
) -> Tuple[str, Dict]:
    ordered_indices = sorted_session_indices(entry, preserve_session_order=preserve_session_order)
    if history_sessions > 0:
        selected_indices = ordered_indices[:history_sessions]
    else:
        selected_indices = ordered_indices

    episode = build_episode(entry, selected_indices, omit_question_date=omit_question_date)
    qa_session_num = len(selected_indices) + 1

    sample_dir = runtime_dir / entry["question_id"]
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    (sample_dir / "data").mkdir(parents=True, exist_ok=True)
    result_dir = sample_dir / "results"

    upstream_data_dir = theanine_dir / "resources" / "data"
    upstream_result_dir = theanine_dir / "results" / "memory"
    upstream_data_dir.mkdir(parents=True, exist_ok=True)
    upstream_result_dir.mkdir(parents=True, exist_ok=True)

    episode_filename = f"bridge_{entry['question_id']}.json"
    episode_path = upstream_data_dir / episode_filename
    write_episode_json(episode_path, episode)

    trace = build_trace_stub(
        entry=entry,
        history_sessions=history_sessions,
        preserve_session_order=preserve_session_order,
        omit_question_date=omit_question_date,
        seed=seed,
    )

    if dry_run:
        hypothesis = f"[dry-run] THEANINE would answer question {entry['question_id']}"
        trace["dry_run"] = True
        return hypothesis, trace

    random.seed(seed)
    Summarizer, MemoryConstructor, Theanine = import_theanine_modules(theanine_dir)
    with maybe_silence_stdout(not verbose_upstream):
        summarizer = Summarizer(
            prompt_name="dialogue-summarization.txt",
            model_name=llm_model,
            temperature=temperature,
            data_name=episode_filename,
            result_path=str(upstream_result_dir),
        )
        summary = summarizer.summarize_all_session()
        summarizer.save(summary)

        constructor = MemoryConstructor(
            prompt_name="relation-extraction.txt",
            model_name=llm_model,
            temperature=temperature,
            data_name=episode_filename,
            summary_path="summary.json",
            result_path=str(upstream_result_dir),
        )
        constructor.linking()
        constructor.save()

        theanine = Theanine(
            prompt_refine="timeline-refinement.txt",
            prompt_rg="response-generation.txt",
            model_name=llm_model,
            temperature=temperature,
            data_name=episode_filename,
            summary_path="summary.json",
            linked_memory_path="linked_memory.json",
        )
        result_dict, total_cost = theanine.theanine_all(session_num=qa_session_num)

    local_episode_copy = sample_dir / "data" / episode_filename
    write_episode_json(local_episode_copy, episode)
    for artifact_name in ["summary.json", "linked_memory.json"]:
        artifact_src = upstream_result_dir / artifact_name
        if artifact_src.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_src, result_dir / artifact_name)

    answer_key = f"s{qa_session_num}-t1"
    answer_payload = result_dict.get(answer_key, {})
    query = entry["question"].strip()
    if not omit_question_date:
        query = f"Current date: {entry['question_date']}\n\n{query}"
    evidence_rows: List[EvidenceRow] = []
    for item in answer_payload.get("after_refinement") or []:
        text = " ".join(str(item).split()).strip()
        if text:
            evidence_rows.append(EvidenceRow(text=text, source="theanine_refined_memory"))
    if not evidence_rows:
        for item in answer_payload.get("before_refinement") or []:
            text = " ".join(str(item).split()).strip()
            if text:
                evidence_rows.append(EvidenceRow(text=text, source="theanine_raw_memory"))
    from langchain_openai import ChatOpenAI

    qa_llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=256,
        model_name=llm_model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = qa_llm.invoke(build_unified_qa_messages(query, evidence_rows))
    hypothesis = (response.content or "").strip()
    trace.update(
        {
            "total_cost": total_cost,
            "input_memory_num": answer_payload.get("input_memory_num"),
            "before_refinement": answer_payload.get("before_refinement"),
            "after_refinement": answer_payload.get("after_refinement"),
            "current_dialogue": answer_payload.get("current_dialogue"),
            "runtime_dir": str(sample_dir),
        }
    )
    return hypothesis, trace


def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    loaded_env = load_env_file(env_candidates)
    if loaded_env:
        print(f"Loaded environment from {loaded_env}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    if not args.dry_run:
        ensure_openai_config(args.theanine_dir, api_key)

    dataset = load_dataset(args.longmemeval_file)
    dataset = dataset[args.offset :]
    if args.limit > 0:
        dataset = dataset[: args.limit]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl.exists():
        args.out_jsonl.unlink()
    if args.trace_jsonl and args.trace_jsonl.exists():
        args.trace_jsonl.unlink()

    ok = 0
    fail = 0
    progress = tqdm(dataset, desc="THEANINE->LongMemEval", unit="q")
    for idx, entry in enumerate(progress):
        qid = entry["question_id"]
        try:
            hypothesis, trace = run_theanine_for_entry(
                entry=entry,
                theanine_dir=args.theanine_dir,
                runtime_dir=args.runtime_dir,
                llm_model=args.llm_model,
                temperature=args.temperature,
                history_sessions=args.history_sessions,
                preserve_session_order=args.preserve_session_order,
                omit_question_date=args.omit_question_date,
                dry_run=args.dry_run,
                verbose_upstream=args.verbose_upstream,
                seed=args.seed + idx,
            )
            append_jsonl(args.out_jsonl, {"question_id": qid, "hypothesis": hypothesis})
            if args.trace_jsonl:
                append_jsonl(args.trace_jsonl, trace)
            ok += 1
            progress.set_postfix(ok=ok, fail=fail, last=qid)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"FAIL qid={qid}: {exc}")
            if args.trace_jsonl:
                error_trace = build_trace_stub(
                    entry=entry,
                    history_sessions=args.history_sessions,
                    preserve_session_order=args.preserve_session_order,
                    omit_question_date=args.omit_question_date,
                    seed=args.seed + idx,
                )
                error_trace.update(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
                append_jsonl(args.trace_jsonl, error_trace)
            progress.set_postfix(ok=ok, fail=fail, last=qid)
            if args.fail_fast:
                raise

    print(f"Done. ok={ok} fail={fail} total={len(dataset)} out={args.out_jsonl}")
    if args.trace_jsonl:
        print(f"Trace saved to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
