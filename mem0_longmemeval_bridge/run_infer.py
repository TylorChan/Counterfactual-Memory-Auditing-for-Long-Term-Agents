#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_prompt_cache import install_openai_prompt_cache

install_openai_prompt_cache("mem0")

from longmemeval_audit import (
    append_jsonl as append_audit_jsonl,
    build_item_record,
    build_query_record,
    build_write_record,
    derive_audit_paths,
    make_write_id,
    normalize_list,
)
from longmemeval_counterfactual import (
    add_cf_args,
    append_cf_outputs,
    build_cf_specs,
    derive_cf_paths,
    parse_dt,
    summarize_replay_cf,
)
from longmemeval_unified_answer import EvidenceRow, build_unified_qa_messages


AGENT = "mem0"
WRITE_TYPE = "mem0_conversation_pair"
PRIMARY_STAGE = "mem0_search_result"
PRIMARY_SOURCE_FORM = "mem0_memory"


def load_env_file(candidates: Sequence[Path], override: bool = False) -> Optional[Path]:
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
            if override or key not in os.environ:
                os.environ[key] = value
        return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mem0 OSS Python SDK on LongMemEval and export UnifiedQA predictions/traces."
    )
    parser.add_argument(
        "--mem0-dir",
        type=Path,
        default=REPO_ROOT / "mem0",
        help="Optional mem0 repo checkout. If missing, import installed mem0ai package.",
    )
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--runtime-storage", type=Path, default=Path("./mem0_longmemeval_bridge/runtime_storage"))
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--embedding-dims", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--search-threshold", type=float, default=0.1)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--omit-question-date", action="store_true")
    parser.add_argument("--preserve-session-order", action="store_true")
    parser.add_argument("--mem0-no-infer", action="store_true", help="Store raw messages instead of mem0's native extraction/update pipeline.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_cf_args(parser)
    return parser.parse_args()


def import_mem0(mem0_dir: Path):
    if mem0_dir and mem0_dir.exists():
        if str(mem0_dir) not in sys.path:
            sys.path.insert(0, str(mem0_dir))
    try:
        from mem0 import Memory  # pylint: disable=import-error
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import mem0. Install with `pip install mem0ai` or pass --mem0-dir to a mem0 repo checkout."
        ) from exc
    return Memory


class OpenAITextClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        from openai import OpenAI

        kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict], retries: int = 2) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"LLM call failed: {last_error}")


def load_longmemeval(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON dataset, got {type(data)} from {path}")
    return data


def parse_longmemeval_datetime(raw: object) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    pattern = r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:\s*\([^)]*\))?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
    match = re.match(pattern, text)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
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


def normalize_timestamp(raw: object) -> str:
    dt = parse_longmemeval_datetime(raw)
    if dt is not None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    parsed = parse_dt(raw)
    if parsed is not None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return "1970-01-01 00:00:00"


def sanitize_path_fragment(raw: object) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or ""))
    clean = clean.strip("._")
    return clean or "sample"


def get_ordered_session_entries(entry: Dict, preserve_order: bool) -> List[Tuple[str, str, List[Dict]]]:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])
    pairs = [
        (
            str(session_ids[idx]) if idx < len(session_ids) and session_ids[idx] else f"s{idx + 1}",
            str(date_raw or ""),
            turns or [],
        )
        for idx, (date_raw, turns) in enumerate(zip(dates, sessions))
    ]
    if preserve_order:
        return pairs

    indexed = []
    for idx, (session_id, date_raw, turns) in enumerate(pairs):
        dt = parse_longmemeval_datetime(date_raw)
        indexed.append((idx, dt, session_id, date_raw, turns))
    indexed.sort(key=lambda x: (1 if x[1] is None else 0, x[1] if x[1] is not None else datetime.max, x[0]))
    return [(session_id, date_raw, turns) for _idx, _dt, session_id, date_raw, turns in indexed]


def iter_message_pairs(turns: Sequence[Dict]) -> Iterable[List[Dict[str, str]]]:
    cleaned: List[Dict[str, str]] = []
    for turn in turns:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    for idx in range(0, len(cleaned), 2):
        pair = cleaned[idx : idx + 2]
        if pair:
            yield pair


def event_content(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages).strip()


def collect_mem0_write_events(entry: Dict, preserve_session_order: bool) -> List[Dict]:
    events: List[Dict] = []
    pair_index = 0
    qid = str(entry["question_id"])
    for session_id, session_date, session_turns in get_ordered_session_entries(entry, preserve_session_order):
        normalized_session_ts = normalize_timestamp(session_date)
        for messages in iter_message_pairs(session_turns):
            pair_index += 1
            content = event_content(messages)
            write_id = make_write_id(
                agent=AGENT,
                question_id=qid,
                write_type=WRITE_TYPE,
                content=content,
                session_id=session_id,
                turn_span=[pair_index],
                timestamp=normalized_session_ts,
            )
            events.append(
                {
                    "write_id": write_id,
                    "session_id": session_id,
                    "turn_span": [pair_index],
                    "timestamp": normalized_session_ts,
                    "messages": messages,
                    "content_text": content,
                    "original_index": pair_index - 1,
                }
            )
    return events


def sort_mem0_events(events: Iterable[Dict]) -> List[Dict]:
    def sort_key(event: Dict):
        dt = parse_dt(event.get("timestamp"))
        if dt is None:
            return (1, str(event.get("timestamp") or ""), int(event.get("original_index", 0)))
        return (0, dt, int(event.get("original_index", 0)))

    return sorted(list(events), key=sort_key)


def apply_mem0_cf_spec(events: Iterable[Dict], spec) -> Tuple[List[Dict], Optional[str]]:
    mutated: List[Dict] = []
    target_timestamp: Optional[str] = None
    for event in events:
        if event["write_id"] != spec.target_write_id:
            mutated.append(dict(event))
            continue
        if spec.cf_type == "rollback":
            target_timestamp = event.get("timestamp")
            continue
        updated = dict(event)
        updated["timestamp"] = spec.new_timestamp or event.get("timestamp")
        target_timestamp = updated["timestamp"]
        mutated.append(updated)
    return sort_mem0_events(mutated), target_timestamp


def build_mem0_event_write_records(qid: str, events: Iterable[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for write_order, event in enumerate(sort_mem0_events(events), start=1):
        records.append(
            build_write_record(
                agent=AGENT,
                question_id=qid,
                write_id=event["write_id"],
                write_order=write_order,
                write_type=WRITE_TYPE,
                stage="write_ingress",
                timestamp=event.get("timestamp"),
                session_id=event.get("session_id"),
                turn_span=event.get("turn_span"),
                content_text=event.get("content_text") or event_content(event.get("messages") or []),
                lineage_source_ids=[event.get("session_id")] if event.get("session_id") else [],
                audit_eligible=True,
                origin="native_memory",
            )
        )
    return records


def build_mem0_memory(args: argparse.Namespace, runtime_dir: Path, collection_name: str):
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    Memory = import_mem0(args.mem0_dir)
    key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --openai-api-key.")
    os.environ["OPENAI_API_KEY"] = key
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url

    qdrant_path = runtime_dir / "qdrant"
    history_db_path = runtime_dir / "history.db"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.llm_model,
                "temperature": 0.0,
                "api_key": key,
                "openai_base_url": args.openai_base_url,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": args.embedding_model,
                "api_key": key,
                "openai_base_url": args.openai_base_url,
                "embedding_dims": args.embedding_dims,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": str(qdrant_path),
                "embedding_model_dims": args.embedding_dims,
                "on_disk": False,
            },
        },
        "history_db_path": str(history_db_path),
        "version": "v1.1",
    }
    return Memory.from_config(config)


def mem0_add(memory, *, messages: List[Dict[str, str]], user_id: str, metadata: Dict, infer: bool) -> Dict:
    return memory.add(messages=messages, user_id=user_id, metadata=metadata, infer=infer) or {"results": []}


def mem0_search(memory, *, query: str, user_id: str, top_k: int, threshold: float, rerank: bool) -> Dict:
    try:
        result = memory.search(query, filters={"user_id": user_id}, top_k=top_k, threshold=threshold, rerank=rerank)
    except TypeError:
        result = memory.search(query, user_id=user_id, limit=top_k)
    if isinstance(result, dict):
        results = result.get("results") or []
        return {"results": results, "raw": result}
    if isinstance(result, list):
        return {"results": result, "raw": {"results": result}}
    return {"results": [], "raw": result}


def result_metadata(result: Dict) -> Dict:
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata


def source_ids_for_result(result: Dict, native_to_source: Dict[str, List[str]]) -> List[str]:
    metadata = result_metadata(result)
    source_ids = normalize_list(metadata.get("source_write_ids") or [])
    if not source_ids and metadata.get("source_write_id"):
        source_ids = normalize_list([metadata.get("source_write_id")])
    native_id = str(result.get("id") or "").strip()
    if not source_ids and native_id:
        source_ids = normalize_list(native_to_source.get(native_id))
    return source_ids


def build_evidence_rows(search_results: Sequence[Dict]) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for result in sorted(search_results, key=lambda item: str(item.get("created_at") or "")):
        memory_text = str(result.get("memory") or result.get("data") or "").strip()
        if not memory_text:
            continue
        rows.append(EvidenceRow(text=memory_text, source=PRIMARY_SOURCE_FORM, timestamp=result.get("created_at")))
    return rows


def run_mem0_replay(
    *,
    entry: Dict,
    events: List[Dict],
    args: argparse.Namespace,
    sample_storage_root: Path,
    sample_tag: str,
) -> Dict:
    qid = str(entry["question_id"])
    qtype = str(entry.get("question_type") or "unknown")
    sample_storage = Path(
        tempfile.mkdtemp(prefix=f"{sample_tag}_{sanitize_path_fragment(qid)}_", dir=str(sample_storage_root))
    )
    collection_name = f"mem0_{sanitize_path_fragment(sample_tag)}_{sanitize_path_fragment(qid)}"[:63]
    user_id = f"longmemeval_{sanitize_path_fragment(qid)}_{sanitize_path_fragment(sample_tag)}"[:128]
    memory = build_mem0_memory(args, sample_storage, collection_name)

    native_to_source: Dict[str, List[str]] = {}
    add_results: List[Dict] = []
    n_ingested_pairs = 0
    infer = not args.mem0_no_infer
    for event in sort_mem0_events(events):
        metadata = {
            "created_at": event.get("timestamp"),
            "source_write_id": event["write_id"],
            "source_write_ids": [event["write_id"]],
            "source_session_id": event.get("session_id"),
            "source_turn_span": event.get("turn_span") or [],
            "source_event_order": event.get("original_index"),
            "question_id": qid,
        }
        response = mem0_add(
            memory,
            messages=event["messages"],
            user_id=user_id,
            metadata=metadata,
            infer=infer,
        )
        results = response.get("results") if isinstance(response, dict) else []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                native_id = str(item.get("id") or "").strip()
                if native_id:
                    native_to_source[native_id] = [event["write_id"]]
                add_results.append({**item, "source_write_id": event["write_id"]})
        n_ingested_pairs += 1

    query = str(entry.get("question") or "")
    if not args.omit_question_date and entry.get("question_date"):
        query = f"Current date: {entry['question_date']}\n\n{query}"
    search_payload = mem0_search(
        memory,
        query=query,
        user_id=user_id,
        top_k=args.top_k,
        threshold=args.search_threshold,
        rerank=args.rerank,
    )
    search_results = [item for item in search_payload.get("results") or [] if isinstance(item, dict)]
    evidence_rows = build_evidence_rows(search_results)
    answer_client = OpenAITextClient(
        api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY") or "",
        model=args.llm_model,
        base_url=args.openai_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    hypothesis = answer_client.chat(build_unified_qa_messages(query, evidence_rows)).strip()

    write_records = build_mem0_event_write_records(qid, events)
    write_record_by_id = {record["write_id"]: record for record in write_records}
    candidate_write_ids = [record["write_id"] for record in write_records]
    retrieved_write_ids: List[str] = []
    prompt_write_ids: List[str] = []
    retrieved_items: List[Dict] = []
    prompt_items: List[Dict] = []
    bridge_items: List[Dict] = []
    seen_item_keys = set()

    def make_item(result: Dict, rank: int, stage: str) -> Optional[Dict]:
        source_ids = source_ids_for_result(result, native_to_source)
        memory_text = str(result.get("memory") or result.get("data") or "").strip()
        native_id = str(result.get("id") or "").strip()
        if not source_ids:
            bridge_items.append(
                {
                    "text": memory_text,
                    "source": "mem0_unmapped_memory",
                    "source_form": PRIMARY_SOURCE_FORM,
                    "native_memory_id": native_id,
                    "audit_eligible": False,
                }
            )
            return None
        dedupe_key = (stage, tuple(source_ids), native_id, memory_text)
        if dedupe_key in seen_item_keys:
            return None
        seen_item_keys.add(dedupe_key)
        source_session_ids = normalize_list(
            write_record_by_id.get(write_id, {}).get("session_id") for write_id in source_ids
        )
        event_timestamps = [
            write_record_by_id.get(write_id, {}).get("timestamp")
            for write_id in source_ids
            if write_record_by_id.get(write_id, {}).get("timestamp")
        ]
        return build_item_record(
            write_id=source_ids[0] if len(source_ids) == 1 else None,
            source_write_ids=source_ids,
            source_session_ids=source_session_ids,
            event_timestamps=event_timestamps,
            memory_timestamps=[result.get("created_at")] if result.get("created_at") else event_timestamps,
            stage=stage,
            rank=rank,
            score=result.get("score"),
            timestamp=result.get("created_at"),
            write_type=WRITE_TYPE,
            source_form=PRIMARY_SOURCE_FORM,
            audit_eligible=True,
            text=memory_text,
            source=stage,
            extra={
                "native_memory_id": native_id,
                "native_event": result.get("event"),
                "metadata": result_metadata(result),
            },
        )

    for rank, result in enumerate(search_results, start=1):
        item_record = make_item(result, rank, PRIMARY_STAGE)
        if item_record is None:
            continue
        retrieved_items.append(dict(item_record))
        retrieved_write_ids.extend(item_record.get("source_write_ids") or [])

    prompt_rank = 0
    for result in sorted(search_results, key=lambda item: str(item.get("created_at") or "")):
        prompt_rank += 1
        item_record = make_item(result, prompt_rank, "mem0_prompt_memory")
        if item_record is None:
            continue
        prompt_items.append(dict(item_record))
        prompt_write_ids.extend(item_record.get("source_write_ids") or [])

    query_record = build_query_record(
        agent=AGENT,
        question_id=qid,
        question_type=qtype,
        query_time=entry.get("question_date"),
        question_date_used=entry.get("question_date"),
        baseline_answer=hypothesis,
        candidate_write_ids=candidate_write_ids,
        retrieved_write_ids=retrieved_write_ids,
        selected_write_ids=prompt_write_ids,
        prompt_write_ids=prompt_write_ids,
        retrieved_items=retrieved_items,
        prompt_items=prompt_items,
        bridge_items=bridge_items,
        extra={
            "query_used": query,
            "mem0_backend": "oss_python_sdk",
            "mem0_infer": infer,
            "mem0_top_k": args.top_k,
            "mem0_search_threshold": args.search_threshold,
            "mem0_rerank": args.rerank,
            "mem0_user_id": user_id,
            "mem0_collection_name": collection_name,
        },
    )
    trace_obj = {
        "question_id": qid,
        "question_type": qtype,
        "n_ingested_pairs": n_ingested_pairs,
        "n_add_results": len(add_results),
        "n_search_results": len(search_results),
        "mem0_user_id": user_id,
        "mem0_collection_name": collection_name,
        "add_results": add_results,
        "search_results": search_results,
    }
    return {
        "hypothesis": hypothesis,
        "trace": trace_obj,
        "query_record": query_record,
        "write_records": write_records,
        "events": sort_mem0_events(events),
    }


def rollback_only(specs: Sequence) -> List:
    return [spec for spec in specs if getattr(spec, "rule_id", "") == "rollback_skip"]


def run_entry_baseline_and_cf(
    *,
    entry: Dict,
    args: argparse.Namespace,
    sample_storage_root: Path,
    sample_index: int,
) -> Dict:
    events = collect_mem0_write_events(entry, args.preserve_session_order)
    outcome = run_mem0_replay(
        entry=entry,
        events=events,
        args=args,
        sample_storage_root=sample_storage_root,
        sample_tag=f"b_{sample_index:03d}",
    )
    if not args.enable_cf_wrapper:
        return outcome

    specs = rollback_only(
        build_cf_specs(
            question_type=str(entry.get("question_type") or "unknown"),
            query_record=outcome["query_record"],
            write_records=outcome["write_records"],
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        )
    )
    cf_results = []
    for spec in specs:
        mutated_events, target_timestamp = apply_mem0_cf_spec(outcome["events"], spec)
        cf_outcome = run_mem0_replay(
            entry=entry,
            events=mutated_events,
            args=args,
            sample_storage_root=sample_storage_root,
            sample_tag=f"cf_{sample_index:03d}_{spec.target_write_id.split(':')[-1]}",
        )
        cf_results.append(
            {
                "spec": spec,
                "cf_answer": cf_outcome["hypothesis"],
                "cf_retrieved_write_ids": cf_outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": cf_outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
                "cf_extra": {
                    "n_search_results": cf_outcome["trace"].get("n_search_results"),
                    "n_add_results": cf_outcome["trace"].get("n_add_results"),
                },
            }
        )
    run_records, query_summary = summarize_replay_cf(
        agent=AGENT,
        entry=entry,
        baseline_query_record=outcome["query_record"],
        write_records=outcome["write_records"],
        cf_results=cf_results,
        dominance_threshold=args.cf_dominance_threshold,
    )
    outcome["cf_run_records"] = run_records
    outcome["cf_query_summary"] = query_summary
    return outcome


def main() -> None:
    args = parse_args()
    loaded_env = load_env_file(
        [
            Path.cwd() / ".env",
            REPO_ROOT / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
    )
    if loaded_env:
        print(f"Loaded environment from {loaded_env}")

    args.runtime_storage.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_jsonl:
        args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_query_path, audit_write_path = derive_audit_paths(args.trace_jsonl)
    cf_run_path, cf_query_path = derive_cf_paths(args.trace_jsonl)
    for path in (audit_query_path, audit_write_path, cf_run_path, cf_query_path):
        if path is not None and path.exists():
            path.unlink()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    dataset = load_longmemeval(args.longmemeval_file)
    if args.offset:
        dataset = dataset[args.offset :]
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    per_sample_storage = args.runtime_storage / "per_sample_replay"
    per_sample_storage.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(dataset)} samples from {args.longmemeval_file}")
    start = time.time()
    n_ok = 0
    n_failed = 0

    with args.out_jsonl.open("w", encoding="utf-8") as pred_f:
        trace_f = args.trace_jsonl.open("w", encoding="utf-8") if args.trace_jsonl else None
        try:
            pbar = tqdm(dataset, total=len(dataset), desc="mem0->LongMemEval", unit="q")
            for i, entry in enumerate(pbar, start=1):
                qid = str(entry.get("question_id", f"idx_{i}"))
                try:
                    if args.dry_run:
                        events = collect_mem0_write_events(entry, args.preserve_session_order)
                        write_records = build_mem0_event_write_records(qid, events)
                        hypothesis = "DRY_RUN_PLACEHOLDER"
                        query_record = build_query_record(
                            agent=AGENT,
                            question_id=qid,
                            question_type=str(entry.get("question_type") or "unknown"),
                            query_time=entry.get("question_date"),
                            question_date_used=entry.get("question_date"),
                            baseline_answer=hypothesis,
                            candidate_write_ids=[record["write_id"] for record in write_records],
                            retrieved_write_ids=[],
                            selected_write_ids=[],
                            prompt_write_ids=[],
                            retrieved_items=[],
                            prompt_items=[],
                            bridge_items=[],
                            extra={"dry_run": True},
                        )
                        outcome = {"hypothesis": hypothesis, "trace": {"question_id": qid, "dry_run": True}, "query_record": query_record, "write_records": write_records}
                    else:
                        outcome = run_entry_baseline_and_cf(
                            entry=entry,
                            args=args,
                            sample_storage_root=per_sample_storage,
                            sample_index=i,
                        )
                        hypothesis = outcome["hypothesis"]

                    pred_f.write(json.dumps({"question_id": qid, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
                    pred_f.flush()
                    if trace_f is not None:
                        trace_f.write(json.dumps(outcome["trace"], ensure_ascii=False) + "\n")
                        trace_f.flush()
                    append_audit_jsonl(audit_query_path, outcome["query_record"])
                    for write_record in outcome.get("write_records") or []:
                        append_audit_jsonl(audit_write_path, write_record)
                    if args.enable_cf_wrapper and not args.dry_run:
                        append_cf_outputs(
                            run_path=cf_run_path,
                            query_path=cf_query_path,
                            run_records=outcome.get("cf_run_records") or [],
                            query_summary=outcome.get("cf_query_summary") or {},
                        )

                    n_ok += 1
                    elapsed = time.time() - start
                    pbar.set_postfix(ok=n_ok, fail=n_failed, last=qid, elapsed_s=f"{elapsed:.1f}")
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    pred_f.write(json.dumps({"question_id": qid, "hypothesis": f"ERROR: {exc}"}, ensure_ascii=False) + "\n")
                    pred_f.flush()
                    tqdm.write(f"FAIL qid={qid}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            if trace_f is not None:
                trace_f.close()

    total = time.time() - start
    print(f"Done. success={n_ok} failed={n_failed} total={len(dataset)} time={total:.1f}s out={args.out_jsonl}")
    if args.trace_jsonl:
        print(f"Trace saved to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
