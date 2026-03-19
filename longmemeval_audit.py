from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRACE_VERSION = "audit_v1"


def append_jsonl(path: Optional[Path], obj: Dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def derive_audit_paths(trace_path: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if trace_path is None:
        return None, None
    stem = trace_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    query_path = trace_path.with_name(f"{stem}.audit_queries.jsonl")
    write_path = trace_path.with_name(f"{stem}.audit_writes.jsonl")
    return query_path, write_path


def normalize_list(items: Optional[Iterable[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def make_content_hash(text: str) -> str:
    payload = " ".join((text or "").split()).strip()
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def make_write_id(
    agent: str,
    question_id: str,
    write_type: str,
    content: str,
    session_id: Optional[str] = None,
    turn_span: Optional[Sequence[object]] = None,
    timestamp: Optional[object] = None,
    origin: str = "native_memory",
) -> str:
    payload = {
        "agent": agent,
        "question_id": question_id,
        "write_type": write_type,
        "content_hash": make_content_hash(content),
        "session_id": session_id or "",
        "turn_span": list(turn_span or []),
        "timestamp": str(timestamp or ""),
        "origin": origin,
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{agent}:{question_id}:{write_type}:{digest}"


def build_write_record(
    *,
    agent: str,
    question_id: str,
    write_id: str,
    write_order: int,
    write_type: str,
    stage: str,
    timestamp: Optional[object],
    session_id: Optional[str],
    turn_span: Optional[Sequence[object]],
    content_text: str,
    lineage_source_ids: Optional[Iterable[str]] = None,
    parent_write_ids: Optional[Iterable[str]] = None,
    audit_eligible: bool = True,
    origin: str = "native_memory",
) -> Dict:
    return {
        "trace_version": TRACE_VERSION,
        "trace_kind": "baseline_write",
        "agent": agent,
        "question_id": question_id,
        "write_id": write_id,
        "write_order": write_order,
        "write_type": write_type,
        "stage": stage,
        "timestamp": None if timestamp is None else str(timestamp),
        "session_id": session_id,
        "turn_span": list(turn_span or []),
        "content_text": content_text,
        "content_hash": make_content_hash(content_text),
        "lineage_source_ids": normalize_list(lineage_source_ids),
        "parent_write_ids": normalize_list(parent_write_ids),
        "audit_eligible": audit_eligible,
        "origin": origin,
    }


def build_query_record(
    *,
    agent: str,
    question_id: str,
    question_type: str,
    query_time: Optional[object],
    question_date_used: Optional[object],
    baseline_answer: str,
    candidate_write_ids: Iterable[str],
    retrieved_write_ids: Iterable[str],
    selected_write_ids: Iterable[str],
    prompt_write_ids: Iterable[str],
    retrieved_items: Optional[List[Dict]] = None,
    prompt_items: Optional[List[Dict]] = None,
    bridge_items: Optional[List[Dict]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    record = {
        "trace_version": TRACE_VERSION,
        "trace_kind": "baseline_query",
        "agent": agent,
        "question_id": question_id,
        "question_type": question_type,
        "query_time": None if query_time is None else str(query_time),
        "question_date_used": None if question_date_used is None else str(question_date_used),
        "baseline_answer": baseline_answer,
        "candidate_write_ids": normalize_list(candidate_write_ids),
        "retrieved_write_ids": normalize_list(retrieved_write_ids),
        "selected_write_ids": normalize_list(selected_write_ids),
        "prompt_write_ids": normalize_list(prompt_write_ids),
        "retrieved_items": retrieved_items or [],
        "prompt_items": prompt_items or [],
        "bridge_items": bridge_items or [],
    }
    if extra:
        record.update(extra)
    return record
