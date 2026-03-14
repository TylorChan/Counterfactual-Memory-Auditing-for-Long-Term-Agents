#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longmemeval_unified_answer import EvidenceRow, build_unified_qa_messages


def load_env_file(candidates: List[Path]) -> Optional[Path]:
    """
    Lightweight .env loader with no external dependency.
    Existing environment variables are not overridden.
    """
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
        description="Run MemoryOS on LongMemEval and export predictions JSONL."
    )
    parser.add_argument(
        "--memoryos-dir",
        type=Path,
        required=True,
        help="Path to MemoryOS repository root (contains memoryos-pypi/).",
    )
    parser.add_argument(
        "--longmemeval-file",
        type=Path,
        required=True,
        help="Path to LongMemEval data JSON (oracle/s_cleaned/etc).",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        required=True,
        help="Output predictions JSONL (question_id + hypothesis).",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=None,
        help="Optional trace JSONL (retrieval diagnostics for later analysis).",
    )
    parser.add_argument(
        "--runtime-storage",
        type=Path,
        default=Path("./memos_longmemeval_bridge/runtime_storage"),
        help="MemoryOS runtime data directory.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--short-term-capacity", type=int, default=7)
    parser.add_argument(
        "--retrieval-queue-capacity",
        type=int,
        default=7,
        help="Retriever page queue capacity (paper/repo setting is 7).",
    )
    parser.add_argument("--mid-term-capacity", type=int, default=2000)
    parser.add_argument("--long-term-knowledge-capacity", type=int, default=100)
    parser.add_argument(
        "--mid-term-heat-threshold",
        type=float,
        default=5.0,
        help="Mid-term heat threshold (paper/repo default is 5).",
    )
    parser.add_argument("--mid-term-similarity-threshold", type=float, default=0.6)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed sample.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call MemoryOS/LLM. Only validate data flow and emit placeholders.",
    )
    parser.add_argument(
        "--response-temperature",
        type=float,
        default=None,
        help="Optional override for all MemoryOS LLM calls. Omit to keep native defaults.",
    )
    parser.add_argument(
        "--response-max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override for all MemoryOS LLM calls.",
    )
    parser.add_argument(
        "--omit-question-date",
        action="store_true",
        help="If set, do not prepend 'Current date' to the final query.",
    )
    parser.add_argument(
        "--preserve-session-order",
        action="store_true",
        help="If set, keep dataset session order; default behavior replays in chronological order.",
    )
    parser.add_argument(
        "--verbose-memoryos",
        action="store_true",
        help="Show verbose logs emitted by MemoryOS internals.",
    )
    parser.add_argument(
        "--reset-mode",
        type=str,
        choices=("reinit", "manual"),
        default="reinit",
        help=(
            "How to isolate samples: 'reinit' rebuilds official Memoryos each question (max repo alignment); "
            "'manual' reuses one instance and clears internal state."
        ),
    )
    return parser.parse_args()


def import_memoryos(memoryos_repo: Path):
    module_root = memoryos_repo / "memoryos-pypi"
    if not module_root.exists():
        raise FileNotFoundError(f"Not found: {module_root}")
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    from memoryos import Memoryos  # pylint: disable=import-error

    return Memoryos


def load_longmemeval(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON dataset, got {type(data)} from {path}")
    return data


def iter_qa_pairs(turns: List[Dict]) -> Iterable[Tuple[str, str]]:
    pending_user: Optional[str] = None
    for turn in turns:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if pending_user is not None:
                yield pending_user, ""
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            yield pending_user, content
            pending_user = None
    if pending_user is not None:
        yield pending_user, ""


def parse_longmemeval_datetime(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    pattern = r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:\s*\([^)]*\))?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
    m = re.match(pattern, raw)
    if not m:
        return None
    year, month, day, hour, minute, second = m.groups()
    try:
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second) if second is not None else 0,
        )
    except ValueError:
        return None


def normalize_timestamp(raw: str) -> str:
    dt = parse_longmemeval_datetime(raw)
    if dt is not None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return "1970-01-01 00:00:00"


@contextmanager
def maybe_silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        yield


def get_ordered_sessions(entry: Dict, preserve_order: bool) -> List[Tuple[str, List[Dict]]]:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    pairs = [(d, s) for d, s in zip(dates, sessions)]
    if preserve_order:
        return pairs

    indexed = []
    for idx, (date_raw, turns) in enumerate(pairs):
        dt = parse_longmemeval_datetime(date_raw)
        indexed.append((idx, dt, date_raw, turns))

    indexed.sort(
        key=lambda x: (
            1 if x[1] is None else 0,
            x[1] if x[1] is not None else datetime.max,
            x[0],
        )
    )
    return [(date_raw, turns) for _idx, _dt, date_raw, turns in indexed]


def reset_memoryos_state(memo) -> None:
    memo.short_term_memory.memory = deque(maxlen=memo.short_term_memory.max_capacity)
    memo.short_term_memory.save()

    memo.mid_term_memory.sessions = {}
    memo.mid_term_memory.access_frequency = defaultdict(int)
    memo.mid_term_memory.heap = []
    memo.mid_term_memory.save()

    memo.user_long_term_memory.user_profiles = {}
    memo.user_long_term_memory.knowledge_base = deque(
        maxlen=memo.user_long_term_memory.knowledge_capacity
    )
    memo.user_long_term_memory.assistant_knowledge = deque(
        maxlen=memo.user_long_term_memory.knowledge_capacity
    )
    memo.user_long_term_memory.save()

    memo.assistant_long_term_memory.user_profiles = {}
    memo.assistant_long_term_memory.knowledge_base = deque(
        maxlen=memo.assistant_long_term_memory.knowledge_capacity
    )
    memo.assistant_long_term_memory.assistant_knowledge = deque(
        maxlen=memo.assistant_long_term_memory.knowledge_capacity
    )
    memo.assistant_long_term_memory.save()

    memo.updater.last_evicted_page_for_continuity = None


def build_memoryos(
    args: argparse.Namespace,
    runtime_storage: Optional[Path] = None,
    user_id: str = "longmemeval_bridge_user",
    assistant_id: str = "longmemeval_bridge_assistant",
):
    Memoryos = import_memoryos(args.memoryos_dir)
    key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --openai-api-key.")

    storage_path = runtime_storage if runtime_storage is not None else args.runtime_storage
    memo = Memoryos(
        user_id=user_id,
        assistant_id=assistant_id,
        openai_api_key=key,
        openai_base_url=args.openai_base_url,
        data_storage_path=str(storage_path),
        llm_model=args.llm_model,
        short_term_capacity=args.short_term_capacity,
        mid_term_capacity=args.mid_term_capacity,
        long_term_knowledge_capacity=args.long_term_knowledge_capacity,
        retrieval_queue_capacity=args.retrieval_queue_capacity,
        mid_term_heat_threshold=args.mid_term_heat_threshold,
        mid_term_similarity_threshold=args.mid_term_similarity_threshold,
        embedding_model_name=args.embedding_model,
    )

    bridge_state = {"last_retrieval": None}
    original_retrieve = memo.retriever.retrieve_context

    def wrapped_retrieve_context(*w_args, **w_kwargs):
        result = original_retrieve(*w_args, **w_kwargs)
        bridge_state["last_retrieval"] = result
        return result

    memo.retriever.retrieve_context = wrapped_retrieve_context

    if args.response_temperature is not None or args.response_max_tokens is not None:
        original_chat_completion = memo.client.chat_completion

        def wrapped_chat_completion(model, messages, temperature=0.7, max_tokens=2000):
            return original_chat_completion(
                model=model,
                messages=messages,
                temperature=(
                    args.response_temperature
                    if args.response_temperature is not None
                    else temperature
                ),
                max_tokens=(
                    args.response_max_tokens
                    if args.response_max_tokens is not None
                    else max_tokens
                ),
            )

        memo.client.chat_completion = wrapped_chat_completion
    memo._bridge_state = bridge_state
    return memo


def replay_sessions_into_memory(memo, entry: Dict, preserve_session_order: bool) -> int:
    n_pairs = 0
    for session_date, session_turns in get_ordered_sessions(entry, preserve_session_order):
        normalized_session_ts = normalize_timestamp(session_date)
        for user_input, agent_response in iter_qa_pairs(session_turns):
            memo.add_memory(
                user_input=user_input,
                agent_response=agent_response,
                timestamp=normalized_session_ts,
            )
            n_pairs += 1
    return n_pairs


def build_memoryos_evidence_rows(retrieval: Dict) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for page in retrieval.get("retrieved_pages", []):
        text_parts = []
        meta_info = (page.get("meta_info") or "").strip()
        user_input = (page.get("user_input") or "").strip()
        agent_response = (page.get("agent_response") or "").strip()
        if meta_info:
            text_parts.append(meta_info)
        if user_input:
            text_parts.append(f"User: {user_input}")
        if agent_response:
            text_parts.append(f"Assistant: {agent_response}")
        text = "\n".join(text_parts).strip()
        if text:
            rows.append(
                EvidenceRow(
                    text=text,
                    source="memoryos_page",
                    timestamp=page.get("timestamp"),
                )
            )
    for item in retrieval.get("retrieved_user_knowledge", []):
        knowledge = (item.get("knowledge") or "").strip()
        if knowledge:
            rows.append(
                EvidenceRow(
                    text=knowledge,
                    source="memoryos_user_knowledge",
                    timestamp=item.get("timestamp"),
                )
            )
    for item in retrieval.get("retrieved_assistant_knowledge", []):
        knowledge = (item.get("knowledge") or "").strip()
        if knowledge:
            rows.append(
                EvidenceRow(
                    text=knowledge,
                    source="memoryos_assistant_knowledge",
                    timestamp=item.get("timestamp"),
                )
            )
    return rows


def build_short_term_rows(memo) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for qa in memo.short_term_memory.get_all():
        user_input = (qa.get("user_input") or "").strip()
        agent_response = (qa.get("agent_response") or "").strip()
        text_parts = []
        if user_input:
            text_parts.append(f"User: {user_input}")
        if agent_response:
            text_parts.append(f"Assistant: {agent_response}")
        text = "\n".join(text_parts).strip()
        if text:
            rows.append(
                EvidenceRow(
                    text=text,
                    source="memoryos_short_term",
                    timestamp=qa.get("timestamp"),
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    silence_memoryos_logs = not args.verbose_memoryos
    loaded_env = load_env_file(
        [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
    )
    if loaded_env:
        print(f"Loaded environment from {loaded_env}")

    args.runtime_storage.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_jsonl:
        args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_longmemeval(args.longmemeval_file)
    if args.offset:
        dataset = dataset[args.offset :]
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Loaded {len(dataset)} samples from {args.longmemeval_file}")
    memo = None
    per_sample_storage = args.runtime_storage / "per_sample_reinit"
    if not args.dry_run and args.reset_mode == "manual":
        with maybe_silence_stdout(silence_memoryos_logs):
            memo = build_memoryos(args)

    start = time.time()
    n_ok = 0
    n_failed = 0

    with args.out_jsonl.open("w", encoding="utf-8") as pred_f:
        trace_f = (
            args.trace_jsonl.open("w", encoding="utf-8")
            if args.trace_jsonl
            else None
        )
        try:
            pbar = tqdm(dataset, total=len(dataset), desc="MemoryOS->LongMemEval", unit="q")
            for i, entry in enumerate(pbar, start=1):
                qid = entry.get("question_id", f"idx_{i}")
                qtype = entry.get("question_type", "unknown")
                try:
                    if args.dry_run:
                        n_pairs = 0
                        for _date, session_turns in get_ordered_sessions(
                            entry, args.preserve_session_order
                        ):
                            for _user_input, _agent_response in iter_qa_pairs(session_turns):
                                n_pairs += 1
                        hypothesis = "DRY_RUN_PLACEHOLDER"
                    else:
                        with maybe_silence_stdout(silence_memoryos_logs):
                            if args.reset_mode == "reinit":
                                if per_sample_storage.exists():
                                    shutil.rmtree(per_sample_storage)
                                per_sample_storage.mkdir(parents=True, exist_ok=True)
                                memo = build_memoryos(
                                    args=args,
                                    runtime_storage=per_sample_storage,
                                    user_id="longmemeval_bridge_user",
                                    assistant_id="longmemeval_bridge_assistant",
                                )
                            else:
                                reset_memoryos_state(memo)
                            memo._bridge_state["last_retrieval"] = None
                            n_pairs = replay_sessions_into_memory(
                                memo, entry, args.preserve_session_order
                            )
                            query = entry["question"]
                            if not args.omit_question_date and entry.get("question_date"):
                                query = f"Current date: {entry['question_date']}\n\n{entry['question']}"
                            retrieval = memo.retriever.retrieve_context(
                                user_query=query,
                                user_id=memo.user_id,
                            )
                            evidence_rows = build_short_term_rows(memo) + build_memoryos_evidence_rows(
                                retrieval
                            )
                            hypothesis = memo.client.chat_completion(
                                model=args.llm_model,
                                messages=build_unified_qa_messages(query, evidence_rows),
                                temperature=0.0,
                                max_tokens=args.response_max_tokens or 256,
                            ).strip()

                    pred_obj = {"question_id": qid, "hypothesis": hypothesis}
                    pred_f.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_f.flush()

                    if trace_f is not None:
                        retrieval = {}
                        short_term_rows = []
                        if not args.dry_run:
                            retrieval = memo._bridge_state.get("last_retrieval") or {}
                            short_term_rows = build_short_term_rows(memo)
                        trace_obj = {
                            "question_id": qid,
                            "question_type": qtype,
                            "reset_mode": args.reset_mode,
                            "n_ingested_pairs": n_pairs,
                            "n_short_term_items": len(short_term_rows),
                            "n_retrieved_pages": len(retrieval.get("retrieved_pages", [])),
                            "n_retrieved_user_knowledge": len(
                                retrieval.get("retrieved_user_knowledge", [])
                            ),
                            "n_retrieved_assistant_knowledge": len(
                                retrieval.get("retrieved_assistant_knowledge", [])
                            ),
                            "short_term_history": [
                                {
                                    "text": row.text,
                                    "timestamp": row.timestamp,
                                    "source": row.source,
                                }
                                for row in short_term_rows
                            ],
                            "retrieved_pages": retrieval.get("retrieved_pages", []),
                            "retrieved_user_knowledge": retrieval.get(
                                "retrieved_user_knowledge", []
                            ),
                            "retrieved_assistant_knowledge": retrieval.get(
                                "retrieved_assistant_knowledge", []
                            ),
                        }
                        trace_f.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")
                        trace_f.flush()

                    n_ok += 1
                    elapsed = time.time() - start
                    pbar.set_postfix(
                        ok=n_ok,
                        fail=n_failed,
                        last=qid,
                        pairs=n_pairs,
                        elapsed_s=f"{elapsed:.1f}",
                    )
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    err_obj = {"question_id": qid, "hypothesis": f"ERROR: {exc}"}
                    pred_f.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                    pred_f.flush()
                    tqdm.write(f"FAIL qid={qid}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            if trace_f is not None:
                trace_f.close()

    total = time.time() - start
    print(
        f"Done. success={n_ok} failed={n_failed} total={len(dataset)} "
        f"time={total:.1f}s out={args.out_jsonl}"
    )
    if args.trace_jsonl:
        print(f"Trace saved to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
