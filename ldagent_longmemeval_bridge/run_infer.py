#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from openai import BadRequestError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


@dataclass
class RetrievalSnapshot:
    context_memories: List[Dict]
    related_memories: List[Dict]


def build_audit_item_from_memory(
    qid: str,
    item: Dict,
    write_type: str,
    stage: str,
    write_order: int,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    text = (item.get("summary") or item.get("dialog") or "").strip()
    if not text:
        return None, None
    idx = item.get("idx")
    timestamp = item.get("time")
    write_id = make_write_id(
        agent="ldagent",
        question_id=qid,
        write_type=write_type,
        content=text,
        session_id=f"idx:{idx}" if idx is not None else None,
        turn_span=[idx] if idx is not None else None,
        timestamp=timestamp,
    )
    write_record = build_write_record(
        agent="ldagent",
        question_id=qid,
        write_id=write_id,
        write_order=write_order,
        write_type=write_type,
        stage=stage,
        timestamp=timestamp,
        session_id=f"idx:{idx}" if idx is not None else None,
        turn_span=[idx] if idx is not None else None,
        content_text=text,
        lineage_source_ids=[f"idx:{idx}"] if idx is not None else [],
        audit_eligible=True,
        origin="native_memory",
    )
    item_record = {
        "write_id": write_id,
        "stage": stage,
        "rank": write_order,
        "score": item.get("score"),
        "timestamp": None if timestamp is None else str(timestamp),
        "write_type": write_type,
        "audit_eligible": True,
    }
    return write_record, item_record


def load_env_file(candidates: List[Path], override: bool = False) -> Optional[Path]:
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
            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)
        return path
    return None


def parse_longmemeval_datetime(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    pattern = r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:\s*\([^)]*\))?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
    match = re.match(pattern, raw)
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


def to_unix_seconds(raw: str, fallback: float) -> float:
    parsed = parse_longmemeval_datetime(raw)
    if parsed is None:
        return fallback
    return parsed.timestamp()


def get_ordered_session_entries(
    entry: Dict,
    preserve_order: bool,
) -> List[Tuple[str, str, List[Dict]]]:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])
    pairs = [
        (
            str(session_ids[idx]) if idx < len(session_ids) and session_ids[idx] else f"s{idx + 1}",
            date_raw,
            turns,
        )
        for idx, (date_raw, turns) in enumerate(zip(dates, sessions))
    ]

    if not preserve_order:
        indexed = []
        for idx, (session_id, date_raw, turns) in enumerate(pairs):
            dt = parse_longmemeval_datetime(date_raw)
            indexed.append((idx, dt, session_id, date_raw, turns))
        indexed.sort(
            key=lambda item: (
                1 if item[1] is None else 0,
                item[1] if item[1] is not None else datetime.max,
                item[0],
            )
        )
        return [(session_id, date_raw, turns) for _idx, _dt, session_id, date_raw, turns in indexed]

    return pairs


def get_ordered_sessions(entry: Dict, preserve_order: bool) -> List[Tuple[str, List[Dict]]]:
    return [(date_raw, turns) for _session_id, date_raw, turns in get_ordered_session_entries(entry, preserve_order)]


def iter_qa_pairs(turns: List[Dict]) -> Iterable[Tuple[str, str]]:
    if not turns:
        return

    if all(isinstance(item, str) for item in turns):
        for user_turn in turns:
            content = user_turn.strip()
            if content:
                yield content, ""
        return

    pending_user: Optional[str] = None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
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


def convert_seconds_to_full_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    units = [
        ("years", 31536000),
        ("months", 2592000),
        ("days", 86400),
        ("hours", 3600),
        ("minutes", 60),
    ]
    parts = []
    for name, count in units:
        value, seconds = divmod(seconds, count)
        if value:
            parts.append(f"{value} {name}")
    return " ".join(parts) if parts else "0 minutes"


def summarize_related_memories(
    related_memories: List[Dict],
    current_time: float,
) -> str:
    if not related_memories:
        return "No relevant Memories."
    summary_lines: List[str] = []
    for item in related_memories:
        item_time = item.get("time", current_time)
        try:
            item_time = float(item_time)
        except (TypeError, ValueError):
            item_time = current_time
        elapsed = convert_seconds_to_full_time(current_time - item_time)
        summary_text = (item.get("summary") or "").strip()
        if not summary_text:
            summary_text = (item.get("dialog") or "").strip()
        if not summary_text:
            continue
        summary_lines.append(f"{elapsed} ago, {summary_text}.")
    return "\n".join(summary_lines) if summary_lines else "No relevant Memories."


def summarize_context(context_memories: List[Dict], user_name: str, inquiry: str) -> str:
    lines: List[str] = []
    for item in context_memories:
        idx = item.get("idx", "")
        dialog = (item.get("dialog") or "").strip()
        if dialog:
            lines.append(f"[TURN {idx}] : {dialog}.")
    lines.append(f"In this turn, {user_name} said: {inquiry}.")
    return "\n".join(lines)


def trim_traits(traits: List[str], max_count: int) -> str:
    if max_count > 0 and len(traits) > max_count:
        return "\n".join(traits[-max_count:])
    return "\n".join(traits)


class OpenAIEmployClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        from openai import OpenAI

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}

    @staticmethod
    def _clean_text(value: object) -> str:
        text = str(value or "")
        text = text.replace("\x00", " ")
        text = "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)
        return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    def _create_with_retry(self, messages: List[Dict]) -> str:
        cleaned_messages = [
            {"role": msg["role"], "content": self._clean_text(msg.get("content", ""))}
            for msg in messages
        ]
        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cleaned_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.token_usage["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.token_usage["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
                    self.token_usage["total"] += int(getattr(usage, "total_tokens", 0) or 0)
                message = response.choices[0].message.content
                return (message or "").strip()
            except BadRequestError as exc:
                last_exc = exc
                if "parse the JSON body" not in str(exc):
                    raise
                time.sleep(1 + attempt)
        raise last_exc

    def employ(self, system_prompt: str, user_prompt: str, name: str = "default") -> str:
        return self._create_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    def chat(self, messages: List[Dict]) -> str:
        return self._create_with_retry(messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LD-Agent memory pipeline on LongMemEval and export predictions JSONL."
    )
    parser.add_argument(
        "--ld-agent-dir",
        type=Path,
        required=True,
        help="Path to LD-Agent repository root (contains Module/).",
    )
    parser.add_argument(
        "--longmemeval-file",
        type=Path,
        required=True,
        help="Path to LongMemEval JSON dataset.",
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
        help="Optional trace JSONL for retrieval diagnostics.",
    )
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-memory-number", type=int, default=30)
    parser.add_argument("--relevance-memory-number", type=int, default=1)
    parser.add_argument("--dist-thres", type=float, default=1.5)
    parser.add_argument(
        "--session-gap-seconds",
        type=int,
        default=3600,
        help="Gap threshold to flush short-term cache into long-term memory (paper uses 600).",
    )
    parser.add_argument("--ori-mem-query", action="store_true")
    parser.add_argument("--max-user-personas", type=int, default=0)
    parser.add_argument("--max-agent-personas", type=int, default=0)
    parser.add_argument("--disable-persona-update", action="store_true")
    parser.add_argument("--omit-question-date", action="store_true")
    parser.add_argument("--preserve-session-order", action="store_true")
    parser.add_argument(
        "--force-flush-before-answer",
        dest="force_flush_before_answer",
        action="store_true",
        default=False,
        help="Force one short->long memory flush after ingest_sessions and before final answer.",
    )
    parser.add_argument(
        "--no-force-flush-before-answer",
        dest="force_flush_before_answer",
        action="store_false",
        help="Disable forced flush before answer (repo-like default behavior).",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--env-override", action="store_true")
    add_cf_args(parser)
    return parser.parse_args()


def build_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("ldagent_longmemeval_bridge")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def import_ldagent_modules(ld_agent_dir: Path):
    module_dir = ld_agent_dir / "Module"
    if not module_dir.exists():
        raise FileNotFoundError(f"Not found: {module_dir}")
    if str(ld_agent_dir) not in sys.path:
        sys.path.insert(0, str(ld_agent_dir))

    from Module.EventMemory import EventMemory  # pylint: disable=import-error
    from Module.Generator import Generator  # pylint: disable=import-error
    from Module.Personas import Personas  # pylint: disable=import-error

    return EventMemory, Personas, Generator


def make_ld_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        usr_name="User",
        agent_name="Assistant",
        max_user_personas=args.max_user_personas,
        max_agent_personas=args.max_agent_personas,
        ori_mem_query=args.ori_mem_query,
        sampling_step=10,
        sampling_path=str(args.out_jsonl.parent),
        sampling_file_name="unused_sampling.json",
    )


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {path}, got {type(data)}")
    return data


def patch_context_retrieve_session_gap(event_memory, session_gap_seconds: int) -> None:
    """
    Patch EventMemory.context_retrieve per instance so session boundary threshold
    can be configured from bridge CLI without editing upstream LD-Agent code.
    """
    gap_seconds = max(0, int(session_gap_seconds))

    def context_retrieve_with_gap(self, query, n_results=10, current_time=0, datatype="text"):
        if (len(self.short_term_memory) > 0) and (
            current_time - self.short_term_memory[-1]["time"]
        ) > gap_seconds:
            last_session_context = [
                f"(line {context_ids + 1}) {context_memory['dialog']}."
                for context_ids, context_memory in enumerate(self.short_term_memory)
            ]
            merged_last_session_context = "\n".join(last_session_context)
            last_session_summary = self.context_summarize(
                merged_last_session_context, len(last_session_context)
            )

            tokenized_item = self.lemma_tokenizer(merged_last_session_context)
            context_nouns_item = list(
                set([token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"])
            )
            merged_nouns_str = ",".join(context_nouns_item)

            metadata = {
                "idx": self.collection.count(),
                "dialog": "",
                "time": self.short_term_memory[-1]["time"],
                "datatype": "text",
                "summary": last_session_summary,
                "topics": merged_nouns_str,
            }
            self.store(self.collection.count(), merged_nouns_str, metadata, datatype="text")

            self.short_term_memory = []
            self.short_term_memory.append(
                {"idx": 0, "time": current_time, "dialog": f"{self.usr_name}: {query}"}
            )
        else:
            self.short_term_memory.append(
                {
                    "idx": len(self.short_term_memory),
                    "time": current_time,
                    "dialog": f"{self.usr_name}: {query}",
                }
            )

        if len(self.short_term_memory) >= n_results:
            return self.short_term_memory[-n_results:]
        return self.short_term_memory

    event_memory.context_retrieve = MethodType(context_retrieve_with_gap, event_memory)


def ingest_sessions(
    entry: Dict,
    event_memory,
    personas,
    ld_args: SimpleNamespace,
    args: argparse.Namespace,
) -> Tuple[int, RetrievalSnapshot]:
    pair_count = 0
    last_context: List[Dict] = []
    last_related: List[Dict] = []

    for session_date, session_turns in get_ordered_sessions(entry, args.preserve_session_order):
        session_ts = to_unix_seconds(session_date, fallback=time.time())
        for user_input, agent_response in iter_qa_pairs(session_turns):
            context_memories = event_memory.context_retrieve(
                user_input,
                n_results=args.context_memory_number,
                current_time=session_ts,
                datatype="text",
            )
            related_memories = event_memory.relevance_retrieve(
                user_input,
                n_results=args.relevance_memory_number,
                dist_thres=args.dist_thres,
                current_time=session_ts,
                datatype="text",
            )

            if not args.disable_persona_update:
                personas.traits_update(user_input, agent_response)

            response_data = {
                "idx": len(event_memory.short_term_memory),
                "time": session_ts,
                "dialog": f"SPEAKER_2: {agent_response}",
            }
            event_memory.short_term_memory.append(response_data)

            pair_count += 1
            last_context = context_memories
            last_related = related_memories

    return pair_count, RetrievalSnapshot(last_context, last_related)


def force_flush_short_term_memory(event_memory) -> int:
    """
    Flush current short-term memory into long-term collection once.
    Returns 1 if flushed, 0 if no-op.
    """
    if len(event_memory.short_term_memory) == 0:
        return 0

    last_session_context = [
        f"(line {context_ids + 1}) {context_memory['dialog']}."
        for context_ids, context_memory in enumerate(event_memory.short_term_memory)
    ]
    merged_last_session_context = "\n".join(last_session_context)
    last_session_summary = event_memory.context_summarize(
        merged_last_session_context, len(last_session_context)
    )

    tokenized_item = event_memory.lemma_tokenizer(merged_last_session_context)
    context_nouns_item = list(
        set([token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"])
    )
    merged_nouns_str = ",".join(context_nouns_item)

    metadata = {
        "idx": event_memory.collection.count(),
        "dialog": "",
        "time": event_memory.short_term_memory[-1]["time"],
        "datatype": "text",
        "summary": last_session_summary,
        "topics": merged_nouns_str,
    }
    event_memory.store(
        event_memory.collection.count(), merged_nouns_str, metadata, datatype="text"
    )

    event_memory.short_term_memory = []
    return 1


def collect_ldagent_write_events(entry: Dict, preserve_session_order: bool) -> List[Dict]:
    events: List[Dict] = []
    pair_idx = 0
    for session_id, session_date, session_turns in get_ordered_session_entries(entry, preserve_session_order):
        session_ts = to_unix_seconds(session_date, fallback=time.time())
        for user_input, agent_response in iter_qa_pairs(session_turns):
            pair_idx += 1
            events.append(
                {
                    "write_id": make_write_id(
                        agent="ldagent",
                        question_id=entry["question_id"],
                        write_type="ldagent_dialogue_turn",
                        content=f"SPEAKER_1: {user_input}\nSPEAKER_2: {agent_response}",
                        session_id=session_id,
                        turn_span=[pair_idx],
                        timestamp=session_ts,
                    ),
                    "session_id": session_id,
                    "turn_span": [pair_idx],
                    "timestamp": session_ts,
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "original_index": pair_idx - 1,
                }
            )
    return events


def sort_ldagent_events(events: Iterable[Dict]) -> List[Dict]:
    def sort_key(event: Dict):
        raw = event.get("timestamp")
        if raw is None:
            return (1, 0.0, int(event.get("original_index", 0)))
        return (0, float(raw), int(event.get("original_index", 0)))

    return sorted(list(events), key=sort_key)


def apply_ldagent_cf_spec(events: Iterable[Dict], spec) -> Tuple[List[Dict], Optional[str]]:
    mutated: List[Dict] = []
    target_timestamp: Optional[str] = None
    for event in events:
        if event["write_id"] != spec.target_write_id:
            mutated.append(dict(event))
            continue
        if spec.cf_type == "rollback":
            target_timestamp = str(event.get("timestamp"))
            continue
        updated = dict(event)
        new_dt = parse_dt(spec.new_timestamp)
        updated["timestamp"] = new_dt.timestamp() if new_dt is not None else event.get("timestamp")
        target_timestamp = spec.new_timestamp
        mutated.append(updated)
    return sort_ldagent_events(mutated), target_timestamp


def build_ldagent_event_write_records(qid: str, events: Iterable[Dict]) -> List[Dict]:
    write_records: List[Dict] = []
    for write_order, event in enumerate(sort_ldagent_events(events), start=1):
        write_records.append(
            build_write_record(
                agent="ldagent",
                question_id=qid,
                write_id=event["write_id"],
                write_order=write_order,
                write_type="ldagent_dialogue_turn",
                stage="write_ingress",
                timestamp=event.get("timestamp"),
                session_id=event.get("session_id"),
                turn_span=event.get("turn_span"),
                content_text=f"SPEAKER_1: {event['user_input']}\nSPEAKER_2: {event['agent_response']}",
                lineage_source_ids=[event.get("session_id")] if event.get("session_id") else [],
                audit_eligible=True,
                origin="native_memory",
            )
        )
    return write_records


def run_ldagent_replay(
    *,
    entry: Dict,
    events: List[Dict],
    args: argparse.Namespace,
    EventMemory,
    Personas,
    Generator,
    llm_client,
    logger,
    ld_args: SimpleNamespace,
) -> Dict:
    qid = entry["question_id"]
    qtype = entry.get("question_type", "unknown")
    event_memory = EventMemory(
        llm_client,
        sample_id=f"cf_longmemeval_{qid}_{int(time.time()*1000)}",
        logger=logger,
        args=ld_args,
        memory_cache=None,
    )
    patch_context_retrieve_session_gap(event_memory, args.session_gap_seconds)
    personas = Personas(llm_client, logger=logger, args=ld_args)
    generator = Generator(
        llm_client,
        sampling_dataset=[],
        sample_id=0,
        logger=logger,
        args=ld_args,
    )

    pair_count = 0
    last_context: List[Dict] = []
    last_related: List[Dict] = []
    event_id_by_idx: Dict[int, str] = {}
    long_term_lineage_by_idx: Dict[int, List[str]] = {}

    original_store = event_memory.store

    def wrapped_store(ids, key, metadata, datatype="text"):
        idx = None
        if isinstance(metadata, dict):
            idx = metadata.get("idx")
        if idx is not None:
            lineage = []
            for item in event_memory.short_term_memory:
                short_idx = item.get("idx")
                write_id = event_id_by_idx.get(short_idx)
                if write_id and write_id not in lineage:
                    lineage.append(write_id)
            if lineage:
                long_term_lineage_by_idx[int(idx)] = lineage
        return original_store(ids, key, metadata, datatype=datatype)

    event_memory.store = wrapped_store
    for event in sort_ldagent_events(events):
        context_memories = event_memory.context_retrieve(
            event["user_input"],
            n_results=args.context_memory_number,
            current_time=event["timestamp"],
            datatype="text",
        )
        related_memories = event_memory.relevance_retrieve(
            event["user_input"],
            n_results=args.relevance_memory_number,
            dist_thres=args.dist_thres,
            current_time=event["timestamp"],
            datatype="text",
        )
        if not args.disable_persona_update:
            personas.traits_update(event["user_input"], event["agent_response"])
        response_data = {
            "idx": len(event_memory.short_term_memory),
            "time": event["timestamp"],
            "dialog": f"SPEAKER_2: {event['agent_response']}",
        }
        event_memory.short_term_memory.append(response_data)
        event_id_by_idx[response_data["idx"]] = event["write_id"]
        pair_count += 1
        last_context = context_memories
        last_related = related_memories

    retrieval_after_ingest = RetrievalSnapshot(last_context, last_related)
    n_forced_flush = 0
    forced_flush_applied = False
    if args.force_flush_before_answer:
        n_forced_flush = force_flush_short_term_memory(event_memory)
        forced_flush_applied = n_forced_flush > 0

    hypothesis, retrieval_for_answer, final_query = answer_question(
        entry,
        event_memory,
        personas,
        generator,
        ld_args,
        args,
    )

    write_records = build_ldagent_event_write_records(qid, events)
    candidate_write_ids = [record["write_id"] for record in write_records]
    write_record_by_id = {record["write_id"]: record for record in write_records}
    retrieved_items: List[Dict] = []
    prompt_items: List[Dict] = []
    retrieved_write_ids: List[str] = []
    prompt_write_ids: List[str] = []
    bridge_items: List[Dict] = []
    seen_ids = set()
    for stage_name, memories in (
        ("answer_context", retrieval_for_answer.context_memories),
        ("answer_related", retrieval_for_answer.related_memories),
    ):
        for item in memories:
            idx = item.get("idx")
            mapped_ids: List[str] = []
            if idx in event_id_by_idx:
                mapped_ids = [event_id_by_idx[idx]]
            elif idx in long_term_lineage_by_idx:
                mapped_ids = list(long_term_lineage_by_idx[idx])
            mapped_ids = normalize_list(mapped_ids)
            evidence_text = item.get("dialog") or item.get("summary") or item.get("text")
            if not mapped_ids:
                bridge_items.append(
                    {
                        "text": evidence_text,
                        "source": stage_name,
                        "source_form": "ldagent_summary",
                        "audit_eligible": False,
                    }
                )
                continue
            dedupe_key = (stage_name, tuple(mapped_ids))
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            source_session_ids = normalize_list(
                write_record_by_id.get(write_id, {}).get("session_id") for write_id in mapped_ids
            )
            event_timestamps = [
                write_record_by_id.get(write_id, {}).get("timestamp")
                for write_id in mapped_ids
                if write_record_by_id.get(write_id, {}).get("timestamp")
            ]
            source_form = "ldagent_raw_dialog" if len(mapped_ids) == 1 else "ldagent_summary"
            item_record = build_item_record(
                write_id=mapped_ids[0] if len(mapped_ids) == 1 else None,
                source_write_ids=mapped_ids,
                source_session_ids=source_session_ids,
                event_timestamps=event_timestamps,
                memory_timestamps=[item.get("time")] if item.get("time") is not None else event_timestamps,
                stage=stage_name,
                rank=len(prompt_items) + 1,
                score=item.get("score"),
                timestamp=item.get("time"),
                write_type="ldagent_dialogue_turn",
                source_form=source_form,
                audit_eligible=True,
                text=evidence_text,
                source=stage_name,
            )
            retrieved_items.append(dict(item_record))
            prompt_items.append(dict(item_record))
            retrieved_write_ids.extend(mapped_ids)
            prompt_write_ids.extend(mapped_ids)

    query_record = build_query_record(
        agent="ldagent",
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
            "query_used": final_query,
            "session_gap_seconds": args.session_gap_seconds,
        },
    )
    trace_obj = {
        "question_id": qid,
        "question_type": qtype,
        "session_gap_seconds": args.session_gap_seconds,
        "ori_mem_query": args.ori_mem_query,
        "dist_thres": args.dist_thres,
        "n_ingested_pairs": pair_count,
        "n_context_after_ingest": len(retrieval_after_ingest.context_memories),
        "n_related_after_ingest": len(retrieval_after_ingest.related_memories),
        "n_context_for_answer": len(retrieval_for_answer.context_memories),
        "n_related_for_answer": len(retrieval_for_answer.related_memories),
        "forced_flush_applied": forced_flush_applied,
        "n_forced_flush": n_forced_flush,
        "query_used": final_query,
        "context_for_answer": retrieval_for_answer.context_memories,
        "related_for_answer": retrieval_for_answer.related_memories,
    }
    return {
        "hypothesis": hypothesis,
        "trace": trace_obj,
        "query_record": query_record,
        "write_records": write_records,
        "events": sort_ldagent_events(events),
    }


def answer_question(
    entry: Dict,
    event_memory,
    personas,
    generator,
    ld_args: SimpleNamespace,
    args: argparse.Namespace,
) -> Tuple[str, RetrievalSnapshot, str]:
    question = entry.get("question", "").strip()
    if not question:
        return "", RetrievalSnapshot([], []), ""

    query_text = question
    if not args.omit_question_date and entry.get("question_date"):
        query_text = f"Current date: {entry['question_date']}\n\n{question}"

    question_ts = to_unix_seconds(entry.get("question_date", ""), fallback=time.time())

    context_memories = event_memory.context_retrieve(
        query_text,
        n_results=args.context_memory_number,
        current_time=question_ts,
        datatype="text",
    )
    related_memories = event_memory.relevance_retrieve(
        query_text,
        n_results=args.relevance_memory_number,
        dist_thres=args.dist_thres,
        current_time=question_ts,
        datatype="text",
    )

    del generator, personas, ld_args

    evidence_rows: List[EvidenceRow] = []
    for item in context_memories:
        dialog = (item.get("dialog") or "").strip()
        if dialog:
            evidence_rows.append(
                EvidenceRow(
                    text=dialog,
                    source="ld_context",
                    timestamp=item.get("time"),
                )
            )
    for item in related_memories:
        summary = (item.get("summary") or item.get("dialog") or "").strip()
        if summary:
            evidence_rows.append(
                EvidenceRow(
                    text=summary,
                    source="ld_related",
                    timestamp=item.get("time"),
                )
            )

    hypothesis = event_memory.LLMclient.chat(build_unified_qa_messages(query_text, evidence_rows)).strip()

    return hypothesis, RetrievalSnapshot(context_memories, related_memories), query_text


def main() -> None:
    args = parse_args()

    loaded_env = load_env_file(
        [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
            Path(__file__).resolve().parent / ".env",
        ],
        override=args.env_override,
    )
    if loaded_env:
        print(f"Loaded environment from {loaded_env}")

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

    dataset = load_dataset(args.longmemeval_file)
    if args.offset:
        dataset = dataset[args.offset:]
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Loaded {len(dataset)} samples from {args.longmemeval_file}")

    if not args.dry_run:
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --openai-api-key.")

        try:
            EventMemory, Personas, Generator = import_ldagent_modules(args.ld_agent_dir)
        except OSError as exc:
            if "en_core_web_sm" in str(exc):
                raise RuntimeError(
                    "spaCy model missing. Install with: python -m spacy download en_core_web_sm"
                ) from exc
            raise

        logger = build_logger(args.verbose)
        ld_args = make_ld_args(args)
        llm_client = OpenAIEmployClient(
            api_key=api_key,
            model=args.llm_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    else:
        EventMemory = Personas = Generator = None
        logger = build_logger(args.verbose)
        ld_args = make_ld_args(args)
        llm_client = None

    start_time = time.time()
    success = 0
    failed = 0

    with args.out_jsonl.open("w", encoding="utf-8") as pred_file:
        trace_file = args.trace_jsonl.open("w", encoding="utf-8") if args.trace_jsonl else None
        try:
            pbar = tqdm(dataset, total=len(dataset), desc="LD-Agent->LongMemEval", unit="q")
            for idx, entry in enumerate(pbar, start=1):
                qid = entry.get("question_id", f"idx_{idx}")
                qtype = entry.get("question_type", "unknown")
                try:
                    if args.dry_run:
                        num_pairs = 0
                        for _session_date, session_turns in get_ordered_sessions(entry, args.preserve_session_order):
                            for _user_input, _agent_response in iter_qa_pairs(session_turns):
                                num_pairs += 1
                        hypothesis = "DRY_RUN_PLACEHOLDER"
                        retrieval_after_ingest = RetrievalSnapshot([], [])
                        retrieval_for_answer = RetrievalSnapshot([], [])
                        final_query = entry.get("question", "")
                        forced_flush_applied = False
                        n_forced_flush = 0
                    else:
                        event_memory = EventMemory(
                            llm_client,
                            sample_id=f"longmemeval_{qid}_{idx}",
                            logger=logger,
                            args=ld_args,
                            memory_cache=None,
                        )
                        patch_context_retrieve_session_gap(
                            event_memory, args.session_gap_seconds
                        )
                        personas = Personas(llm_client, logger=logger, args=ld_args)
                        generator = Generator(
                            llm_client,
                            sampling_dataset=[],
                            sample_id=idx,
                            logger=logger,
                            args=ld_args,
                        )

                        num_pairs, retrieval_after_ingest = ingest_sessions(
                            entry,
                            event_memory,
                            personas,
                            ld_args,
                            args,
                        )

                        n_forced_flush = 0
                        forced_flush_applied = False
                        if args.force_flush_before_answer:
                            n_forced_flush = force_flush_short_term_memory(event_memory)
                            forced_flush_applied = n_forced_flush > 0

                        hypothesis, retrieval_for_answer, final_query = answer_question(
                            entry,
                            event_memory,
                            personas,
                            generator,
                            ld_args,
                            args,
                        )

                    pred_obj = {"question_id": qid, "hypothesis": hypothesis}
                    pred_file.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_file.flush()

                    if trace_file is not None:
                        trace_obj = {
                            "question_id": qid,
                            "question_type": qtype,
                            "session_gap_seconds": args.session_gap_seconds,
                            "ori_mem_query": args.ori_mem_query,
                            "dist_thres": args.dist_thres,
                            "n_ingested_pairs": num_pairs,
                            "n_context_after_ingest": len(retrieval_after_ingest.context_memories),
                            "n_related_after_ingest": len(retrieval_after_ingest.related_memories),
                            "n_context_for_answer": len(retrieval_for_answer.context_memories),
                            "n_related_for_answer": len(retrieval_for_answer.related_memories),
                            "forced_flush_applied": forced_flush_applied,
                            "n_forced_flush": n_forced_flush,
                            "query_used": final_query,
                            "context_for_answer": retrieval_for_answer.context_memories,
                            "related_for_answer": retrieval_for_answer.related_memories,
                        }
                        if not args.dry_run and llm_client is not None:
                            trace_obj["token_usage"] = llm_client.token_usage
                        trace_file.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")
                        trace_file.flush()

                    write_records: List[Dict] = []
                    retrieved_items: List[Dict] = []
                    prompt_items: List[Dict] = []
                    candidate_write_ids: List[str] = []
                    retrieved_write_ids: List[str] = []
                    prompt_write_ids: List[str] = []
                    write_order = 0
                    for stage_name, memories in (
                        ("after_ingest_context", retrieval_after_ingest.context_memories),
                        ("after_ingest_related", retrieval_after_ingest.related_memories),
                        ("answer_context", retrieval_for_answer.context_memories),
                        ("answer_related", retrieval_for_answer.related_memories),
                    ):
                        write_type = "context_memory" if "context" in stage_name else "related_memory"
                        for item in memories:
                            write_order += 1
                            write_record, item_record = build_audit_item_from_memory(
                                qid=qid,
                                item=item,
                                write_type=write_type,
                                stage=stage_name,
                                write_order=write_order,
                            )
                            if write_record is None or item_record is None:
                                continue
                            write_records.append(write_record)
                            candidate_write_ids.append(write_record["write_id"])
                            if stage_name.startswith("answer_"):
                                retrieved_items.append(item_record)
                                prompt_items.append(dict(item_record))
                                retrieved_write_ids.append(write_record["write_id"])
                                prompt_write_ids.append(write_record["write_id"])

                    query_record = build_query_record(
                        agent="ldagent",
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
                        bridge_items=[],
                        extra={
                            "query_used": final_query,
                            "session_gap_seconds": args.session_gap_seconds,
                        },
                    )
                    append_audit_jsonl(audit_query_path, query_record)
                    for write_record in write_records:
                        append_audit_jsonl(audit_write_path, write_record)
                    if args.enable_cf_wrapper and not args.dry_run:
                        cf_events = collect_ldagent_write_events(entry, args.preserve_session_order)
                        cf_baseline = run_ldagent_replay(
                            entry=entry,
                            events=cf_events,
                            args=args,
                            EventMemory=EventMemory,
                            Personas=Personas,
                            Generator=Generator,
                            llm_client=llm_client,
                            logger=logger,
                            ld_args=ld_args,
                        )
                        cf_write_records = cf_baseline["write_records"]
                        cf_query_record = cf_baseline["query_record"]
                        specs = build_cf_specs(
                            question_type=qtype,
                            query_record=cf_query_record,
                            write_records=cf_write_records,
                            answer_session_ids=entry.get("answer_session_ids", []),
                            max_writes=args.cf_max_writes,
                            scope=args.cf_target_scope,
                        )
                        cf_results = []
                        for spec in specs:
                            mutated_events, target_timestamp = apply_ldagent_cf_spec(cf_baseline["events"], spec)
                            cf_outcome = run_ldagent_replay(
                                entry=entry,
                                events=mutated_events,
                                args=args,
                                EventMemory=EventMemory,
                                Personas=Personas,
                                Generator=Generator,
                                llm_client=llm_client,
                                logger=logger,
                                ld_args=ld_args,
                            )
                            cf_results.append(
                                {
                                    "spec": spec,
                                    "cf_answer": cf_outcome["hypothesis"],
                                    "cf_retrieved_write_ids": cf_outcome["query_record"].get("retrieved_write_ids", []),
                                    "cf_prompt_write_ids": cf_outcome["query_record"].get("prompt_write_ids", []),
                                    "target_timestamp": target_timestamp,
                                }
                            )
                        run_records, query_summary = summarize_replay_cf(
                            agent="ldagent",
                            entry=entry,
                            baseline_query_record=cf_query_record,
                            write_records=cf_write_records,
                            cf_results=cf_results,
                            dominance_threshold=args.cf_dominance_threshold,
                        )
                        append_cf_outputs(
                            run_path=cf_run_path,
                            query_path=cf_query_path,
                            run_records=run_records,
                            query_summary=query_summary,
                        )

                    success += 1
                    elapsed = time.time() - start_time
                    pbar.set_postfix(
                        ok=success,
                        fail=failed,
                        last=qid,
                        pairs=num_pairs,
                        elapsed_s=f"{elapsed:.1f}",
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    pred_obj = {"question_id": qid, "hypothesis": f"ERROR: {exc}"}
                    pred_file.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_file.flush()
                    tqdm.write(f"FAIL qid={qid}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            if trace_file is not None:
                trace_file.close()

    total_elapsed = time.time() - start_time
    print(
        f"Done. success={success} failed={failed} total={len(dataset)} "
        f"time={total_elapsed:.1f}s out={args.out_jsonl}"
    )
    if args.trace_jsonl:
        print(f"Trace saved to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
