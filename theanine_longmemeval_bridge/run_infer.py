#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_prompt_cache import install_openai_prompt_cache

install_openai_prompt_cache("theanine")

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

SESSION_NAMES = ["first", "second", "third", "fourth", "fifth"]


class TheanineWriteEvent(dict):
    pass


def _normalize_theanine_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _extract_theanine_summary_segments(text: object) -> List[str]:
    raw = str(text or "")
    bracketed = [
        _normalize_theanine_text(match)
        for match in re.findall(r"\[([^\]]+)\]", raw)
        if _normalize_theanine_text(match)
    ]
    if bracketed:
        return bracketed
    return [_normalize_theanine_text(raw)] if _normalize_theanine_text(raw) else []


def _lookup_theanine_source_matches(
    text: object,
    text_to_records: Dict[str, List[Tuple[str, Dict]]],
) -> List[Tuple[str, Dict]]:
    normalized_text = _normalize_theanine_text(text)
    if not normalized_text:
        return []
    direct = text_to_records.get(normalized_text, [])
    if direct:
        return direct

    aggregated: List[Tuple[str, Dict]] = []
    seen = set()
    for segment in _extract_theanine_summary_segments(text):
        for write_id, item_record in text_to_records.get(segment, []):
            key = (write_id, item_record.get("timestamp"), item_record.get("session_id"))
            if key in seen:
                continue
            seen.add(key)
            aggregated.append((write_id, item_record))
    return aggregated


def _lookup_theanine_source_write_ids(text: object, text_to_write_ids: Dict[str, List[str]]) -> List[str]:
    normalized_text = _normalize_theanine_text(text)
    if not normalized_text:
        return []
    direct = normalize_list(text_to_write_ids.get(normalized_text, []))
    if direct:
        return direct
    source_write_ids: List[str] = []
    for segment in _extract_theanine_summary_segments(text):
        source_write_ids.extend(text_to_write_ids.get(segment, []))
    return normalize_list(source_write_ids)


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


class OpenAIAnswerClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout)
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
    add_cf_args(parser)
    return parser.parse_args()


@contextmanager
def maybe_silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        yield


def ensure_openai_config(config_root: Path, api_key: str) -> Path:
    conf_dir = config_root / "conf.d"
    conf_dir.mkdir(parents=True, exist_ok=True)
    config_path = conf_dir / "config.yaml"
    config_path.write_text("openai:\n  key: ${OPENAI_API_KEY}\n", encoding="utf-8")
    return config_path


def ensure_upstream_workspace(theanine_dir: Path, workspace_dir: Path, api_key: str) -> Dict[str, Path]:
    resources_dir = workspace_dir / "resources"
    prompts_dir = resources_dir / "prompts"
    data_dir = resources_dir / "data"
    result_dir = workspace_dir / "results" / "memory"

    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if prompts_dir.exists() or prompts_dir.is_symlink():
        if prompts_dir.is_symlink() and prompts_dir.resolve() == (theanine_dir / "resources" / "prompts").resolve():
            pass
        else:
            if prompts_dir.is_dir() and not prompts_dir.is_symlink():
                shutil.rmtree(prompts_dir)
            else:
                prompts_dir.unlink()
            prompts_dir.symlink_to(theanine_dir / "resources" / "prompts", target_is_directory=True)
    else:
        prompts_dir.parent.mkdir(parents=True, exist_ok=True)
        prompts_dir.symlink_to(theanine_dir / "resources" / "prompts", target_is_directory=True)

    config_path = ensure_openai_config(workspace_dir, api_key)
    return {
        "workspace_dir": workspace_dir,
        "data_dir": data_dir,
        "result_dir": result_dir,
        "config_path": config_path,
    }


@contextmanager
def theanine_workspace_env(workspace_dir: Path, config_path: Path):
    old_project = os.environ.get("THEANINE_PROJECT_PATH")
    old_config = os.environ.get("THEANINE_CONFIG_PATH")
    os.environ["THEANINE_PROJECT_PATH"] = str(workspace_dir)
    os.environ["THEANINE_CONFIG_PATH"] = str(config_path)
    try:
        yield
    finally:
        if old_project is None:
            os.environ.pop("THEANINE_PROJECT_PATH", None)
        else:
            os.environ["THEANINE_PROJECT_PATH"] = old_project
        if old_config is None:
            os.environ.pop("THEANINE_CONFIG_PATH", None)
        else:
            os.environ["THEANINE_CONFIG_PATH"] = old_config


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


def collect_theanine_write_events(
    entry: Dict,
    history_sessions: int,
    preserve_session_order: bool,
) -> List[TheanineWriteEvent]:
    ordered_indices = sorted_session_indices(entry, preserve_session_order=preserve_session_order)
    if history_sessions > 0:
        ordered_indices = ordered_indices[:history_sessions]
    events: List[TheanineWriteEvent] = []
    for original_index, idx in enumerate(ordered_indices):
        events.append(
            TheanineWriteEvent(
                write_id=make_write_id(
                    agent="theanine",
                    question_id=entry["question_id"],
                    write_type="theanine_session_ingress",
                    content=json.dumps(entry["haystack_sessions"][idx], ensure_ascii=False),
                    session_id=entry["haystack_session_ids"][idx],
                    turn_span=[idx],
                    timestamp=entry["haystack_dates"][idx],
                ),
                session_id=entry["haystack_session_ids"][idx],
                timestamp=entry["haystack_dates"][idx],
                turns=entry["haystack_sessions"][idx],
                source_index=idx,
                original_index=original_index,
            )
        )
    return events


def sort_theanine_events(events: Sequence[TheanineWriteEvent]) -> List[TheanineWriteEvent]:
    def sort_key(event: TheanineWriteEvent):
        dt = parse_dt(event.get("timestamp"))
        if dt is None:
            return (1, str(event.get("timestamp") or ""), int(event.get("original_index", 0)))
        return (0, dt, int(event.get("original_index", 0)))

    return sorted(list(events), key=sort_key)


def apply_theanine_cf_spec(
    events: Sequence[TheanineWriteEvent],
    spec,
) -> Tuple[List[TheanineWriteEvent], Optional[str]]:
    mutated: List[TheanineWriteEvent] = []
    target_timestamp: Optional[str] = None
    for event in events:
        if event["write_id"] != spec.target_write_id:
            mutated.append(TheanineWriteEvent(**dict(event)))
            continue
        if spec.cf_type == "rollback":
            target_timestamp = event.get("timestamp")
            continue
        updated = TheanineWriteEvent(**dict(event))
        updated["timestamp"] = spec.new_timestamp or event.get("timestamp")
        target_timestamp = updated["timestamp"]
        mutated.append(updated)
    return sort_theanine_events(mutated), target_timestamp


def build_episode_from_events(entry: Dict, events: Sequence[TheanineWriteEvent], omit_question_date: bool) -> Dict:
    history_session_count = len(events)
    total_session_count = history_session_count + 1
    episode: Dict[str, object] = {
        "dataID": entry["question_id"],
        "history_session_count": history_session_count,
        "total_session_count": total_session_count,
    }
    for session_num, event in enumerate(events, start=1):
        prefix = session_field_prefix(session_num)
        dialogue, speakers = to_theanine_speakers(event["turns"])
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


def parse_summary_session_num(node_id: str) -> Optional[int]:
    try:
        prefix = node_id.split("-", 1)[0]
        if prefix.startswith("s"):
            return int(prefix[1:])
    except Exception:  # noqa: BLE001
        return None
    return None


def build_theanine_audit_records(entry: Dict, trace: Dict, hypothesis: str) -> Tuple[List[Dict], Dict]:
    runtime_root = trace.get("runtime_dir")
    if not runtime_root:
        return [], build_query_record(
            agent="theanine",
            question_id=entry["question_id"],
            question_type=entry.get("question_type", "unknown"),
            query_time=entry.get("question_date"),
            question_date_used=trace.get("question_date_used"),
            baseline_answer=hypothesis,
            candidate_write_ids=[],
            retrieved_write_ids=[],
            selected_write_ids=[],
            prompt_write_ids=[],
            retrieved_items=[],
            prompt_items=[],
            bridge_items=[],
            extra={
                "history_sessions_requested": trace.get("history_sessions_requested"),
                "history_sessions_used": trace.get("history_sessions_used"),
                "dry_run": trace.get("dry_run", False),
            },
        )

    runtime_dir = Path(runtime_root) / "results"
    summary_path = runtime_dir / "summary.json"
    if not summary_path.exists():
        return [], build_query_record(
            agent="theanine",
            question_id=entry["question_id"],
            question_type=entry.get("question_type", "unknown"),
            query_time=entry.get("question_date"),
            question_date_used=trace.get("question_date_used"),
            baseline_answer=hypothesis,
            candidate_write_ids=[],
            retrieved_write_ids=[],
            selected_write_ids=[],
            prompt_write_ids=[],
            retrieved_items=[],
            prompt_items=[],
            bridge_items=[],
            extra={"query_used": entry.get("question", "")},
        )

    summary_nodes = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_session_dates = trace.get("selected_session_dates", [])
    selected_session_ids = trace.get("selected_session_ids", [])
    session_time_map = {idx + 1: selected_session_dates[idx] for idx in range(len(selected_session_dates))}
    session_id_map = {idx + 1: selected_session_ids[idx] for idx in range(len(selected_session_ids))}

    write_records: List[Dict] = []
    candidate_write_ids: List[str] = []
    text_to_ids: Dict[str, List[Tuple[str, Dict]]] = {}
    for write_order, (node_id, text) in enumerate(summary_nodes.items(), start=1):
        if not isinstance(text, str) or not text.strip():
            continue
        session_num = parse_summary_session_num(node_id)
        timestamp = session_time_map.get(session_num)
        session_id = session_id_map.get(session_num, f"s{session_num}" if session_num is not None else None)
        write_id = make_write_id(
            agent="theanine",
            question_id=entry["question_id"],
            write_type="theanine_summary_node",
            content=text,
            session_id=session_id,
            turn_span=[node_id],
            timestamp=timestamp,
        )
        write_record = build_write_record(
            agent="theanine",
            question_id=entry["question_id"],
            write_id=write_id,
            write_order=write_order,
            write_type="theanine_summary_node",
            stage="summary_memory_state",
            timestamp=timestamp,
            session_id=session_id,
            turn_span=[node_id],
            content_text=text,
            lineage_source_ids=[session_id, node_id] if session_id else [node_id],
            audit_eligible=True,
            origin="native_memory",
        )
        write_records.append(write_record)
        candidate_write_ids.append(write_id)
        item_record = {
            "write_id": write_id,
            "stage": "summary_memory_state",
            "rank": write_order,
            "score": None,
            "timestamp": None if timestamp is None else str(timestamp),
            "write_type": "theanine_summary_node",
            "audit_eligible": True,
        }
        norm_text = " ".join(text.split()).strip()
        text_to_ids.setdefault(norm_text, []).append((write_id, item_record))

    retrieved_write_ids: List[str] = []
    selected_write_ids: List[str] = []
    prompt_write_ids: List[str] = []
    retrieved_items: List[Dict] = []
    prompt_items: List[Dict] = []
    bridge_items: List[Dict] = []
    seen_ids = set()
    before_source_write_ids: List[str] = []
    before_source_session_ids: List[str] = []
    before_source_timestamps: List[str] = []

    for item in trace.get("before_refinement") or []:
        matches = _lookup_theanine_source_matches(item, text_to_ids)
        if not matches:
            bridge_items.append(
                {"text": item, "source": "theanine_before_refinement_unmapped", "audit_eligible": False}
            )
            continue
        source_write_ids = normalize_list(write_id for write_id, _ in matches)
        source_session_ids = normalize_list(item_record.get("session_id") for _write_id, item_record in matches)
        source_timestamps = [
            item_record.get("timestamp")
            for _write_id, item_record in matches
            if item_record.get("timestamp")
        ]
        dedupe_key = ("before_refinement", tuple(source_write_ids))
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        before_source_write_ids.extend(source_write_ids)
        before_source_session_ids.extend(source_session_ids)
        before_source_timestamps.extend(source_timestamps)
        retrieved_write_ids.extend(source_write_ids)
        selected_write_ids.extend(source_write_ids)
        prompt_write_ids.extend(source_write_ids)
        item_record = build_item_record(
            write_id=source_write_ids[0] if len(source_write_ids) == 1 else None,
            source_write_ids=source_write_ids,
            source_session_ids=source_session_ids,
            event_timestamps=source_timestamps,
            memory_timestamps=source_timestamps,
            stage="before_refinement",
            rank=len(prompt_items) + 1,
            score=None,
            timestamp=source_timestamps[0] if source_timestamps else None,
            write_type="theanine_summary_node",
            source_form="theanine_before_refinement",
            audit_eligible=True,
            text=str(item),
            source="before_refinement",
        )
        retrieved_items.append(dict(item_record))
        prompt_items.append(dict(item_record))

    for item in trace.get("after_refinement") or []:
        if before_source_write_ids:
            item_record = build_item_record(
                write_id=before_source_write_ids[0] if len(normalize_list(before_source_write_ids)) == 1 else None,
                source_write_ids=before_source_write_ids,
                source_session_ids=before_source_session_ids,
                event_timestamps=before_source_timestamps,
                memory_timestamps=[trace.get("question_date_used")] if trace.get("question_date_used") else before_source_timestamps,
                stage="after_refinement",
                rank=len(prompt_items) + 1,
                score=None,
                timestamp=trace.get("question_date_used"),
                write_type="theanine_summary_node",
                source_form="theanine_after_refinement",
                audit_eligible=True,
                text=str(item),
                source="after_refinement",
                extra={"parent_write_ids": normalize_list(before_source_write_ids)},
            )
            retrieved_items.append(dict(item_record))
            prompt_items.append(dict(item_record))
        else:
            bridge_items.append(
                {
                    "text": item,
                    "source": "theanine_after_refinement",
                    "source_form": "theanine_after_refinement",
                    "audit_eligible": False,
                }
            )

    query_record = build_query_record(
        agent="theanine",
        question_id=entry["question_id"],
        question_type=entry.get("question_type", "unknown"),
        query_time=entry.get("question_date"),
        question_date_used=trace.get("question_date_used"),
        baseline_answer=hypothesis,
        candidate_write_ids=candidate_write_ids,
        retrieved_write_ids=retrieved_write_ids,
        selected_write_ids=selected_write_ids,
        prompt_write_ids=prompt_write_ids,
        retrieved_items=retrieved_items,
        prompt_items=prompt_items,
        bridge_items=bridge_items,
        extra={
            "history_sessions_requested": trace.get("history_sessions_requested"),
            "history_sessions_used": trace.get("history_sessions_used"),
        },
    )
    return write_records, query_record


def build_theanine_event_write_records(entry: Dict, events: Sequence[TheanineWriteEvent]) -> List[Dict]:
    write_records: List[Dict] = []
    for write_order, event in enumerate(sort_theanine_events(events), start=1):
        write_records.append(
            build_write_record(
                agent="theanine",
                question_id=entry["question_id"],
                write_id=event["write_id"],
                write_order=write_order,
                write_type="theanine_session_ingress",
                stage="write_ingress",
                timestamp=event.get("timestamp"),
                session_id=event.get("session_id"),
                turn_span=[event.get("source_index")],
                content_text=json.dumps(event.get("turns", []), ensure_ascii=False),
                lineage_source_ids=[event.get("session_id")] if event.get("session_id") else [],
                audit_eligible=True,
                origin="native_memory",
            )
        )
    return write_records


def build_theanine_replay_audit(
    entry: Dict,
    events: Sequence[TheanineWriteEvent],
    trace: Dict,
    hypothesis: str,
) -> Tuple[List[Dict], Dict]:
    write_records = build_theanine_event_write_records(entry, events)
    event_ids_by_session: Dict[str, str] = {str(event["session_id"]): event["write_id"] for event in events}
    runtime_root = trace.get("runtime_dir")
    text_to_write_ids: Dict[str, List[str]] = {}
    if runtime_root:
        summary_path = Path(runtime_root) / "results" / "summary.json"
        if summary_path.exists():
            summary_nodes = json.loads(summary_path.read_text(encoding="utf-8"))
            selected_session_ids = trace.get("selected_session_ids") or []
            for node_id, text in summary_nodes.items():
                if not isinstance(text, str) or not text.strip():
                    continue
                session_num = parse_summary_session_num(node_id)
                if session_num is None or session_num < 1 or session_num > len(selected_session_ids):
                    continue
                write_id = event_ids_by_session.get(str(selected_session_ids[session_num - 1]))
                if not write_id:
                    continue
                norm_text = " ".join(text.split()).strip()
                text_to_write_ids.setdefault(norm_text, []).append(write_id)
    retrieved_write_ids: List[str] = []
    selected_write_ids: List[str] = []
    prompt_write_ids: List[str] = []
    retrieved_items: List[Dict] = []
    prompt_items: List[Dict] = []
    bridge_items: List[Dict] = []
    seen = set()
    before_source_write_ids: List[str] = []
    before_source_timestamps: List[str] = []
    for item in trace.get("before_refinement") or []:
        text = _normalize_theanine_text(item)
        if not text:
            continue
        source_write_ids = _lookup_theanine_source_write_ids(item, text_to_write_ids)
        if not source_write_ids:
            bridge_items.append(
                {
                    "text": item,
                    "source": "theanine_before_refinement_unmapped",
                    "source_form": "theanine_before_refinement",
                    "audit_eligible": False,
                }
            )
            continue
        dedupe_key = ("before_refinement", tuple(source_write_ids))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_timestamps = [
            record["timestamp"]
            for record in write_records
            if record["write_id"] in set(source_write_ids) and record.get("timestamp")
        ]
        before_source_write_ids.extend(source_write_ids)
        before_source_timestamps.extend(source_timestamps)
        item_record = build_item_record(
            write_id=source_write_ids[0] if len(source_write_ids) == 1 else None,
            source_write_ids=source_write_ids,
            source_session_ids=[],
            event_timestamps=source_timestamps,
            memory_timestamps=source_timestamps,
            stage="before_refinement",
            rank=len(prompt_items) + 1,
            score=None,
            timestamp=source_timestamps[0] if source_timestamps else None,
            write_type="theanine_session_ingress",
            source_form="theanine_before_refinement",
            audit_eligible=True,
            text=text,
            source="before_refinement",
        )
        retrieved_write_ids.extend(source_write_ids)
        selected_write_ids.extend(source_write_ids)
        prompt_write_ids.extend(source_write_ids)
        retrieved_items.append(dict(item_record))
        prompt_items.append(dict(item_record))
    for item in trace.get("after_refinement") or []:
        if before_source_write_ids:
            item_record = build_item_record(
                write_id=before_source_write_ids[0] if len(normalize_list(before_source_write_ids)) == 1 else None,
                source_write_ids=before_source_write_ids,
                source_session_ids=[],
                event_timestamps=before_source_timestamps,
                memory_timestamps=[trace.get("question_date_used")] if trace.get("question_date_used") else before_source_timestamps,
                stage="after_refinement",
                rank=len(prompt_items) + 1,
                score=None,
                timestamp=trace.get("question_date_used"),
                write_type="theanine_session_ingress",
                source_form="theanine_after_refinement",
                audit_eligible=True,
                text=str(item),
                source="after_refinement",
                extra={"parent_write_ids": normalize_list(before_source_write_ids)},
            )
            retrieved_items.append(dict(item_record))
            prompt_items.append(dict(item_record))
        else:
            bridge_items.append(
                {
                    "text": item,
                    "source": "theanine_after_refinement",
                    "source_form": "theanine_after_refinement",
                    "audit_eligible": False,
                }
            )
    query_record = build_query_record(
        agent="theanine",
        question_id=entry["question_id"],
        question_type=entry.get("question_type", "unknown"),
        query_time=entry.get("question_date"),
        question_date_used=trace.get("question_date_used"),
        baseline_answer=hypothesis,
        candidate_write_ids=[record["write_id"] for record in write_records],
        retrieved_write_ids=retrieved_write_ids,
        selected_write_ids=selected_write_ids,
        prompt_write_ids=prompt_write_ids,
        retrieved_items=retrieved_items,
        prompt_items=prompt_items,
        bridge_items=bridge_items,
        extra={
            "history_sessions_requested": trace.get("history_sessions_requested"),
            "history_sessions_used": trace.get("history_sessions_used"),
        },
    )
    return write_records, query_record


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
    events: Optional[Sequence[TheanineWriteEvent]] = None,
) -> Tuple[str, Dict]:
    if events is None:
        ordered_indices = sorted_session_indices(entry, preserve_session_order=preserve_session_order)
        if history_sessions > 0:
            selected_indices = ordered_indices[:history_sessions]
        else:
            selected_indices = ordered_indices
        episode = build_episode(entry, selected_indices, omit_question_date=omit_question_date)
        selected_session_ids = [entry["haystack_session_ids"][i] for i in selected_indices]
        selected_session_dates = [entry["haystack_dates"][i] for i in selected_indices]
    else:
        selected_indices = [int(event["source_index"]) for event in events]
        episode = build_episode_from_events(entry, events, omit_question_date=omit_question_date)
        selected_session_ids = [str(event["session_id"]) for event in events]
        selected_session_dates = [str(event["timestamp"]) for event in events]
    qa_session_num = len(selected_indices) + 1

    sample_dir = runtime_dir / entry["question_id"]
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    (sample_dir / "data").mkdir(parents=True, exist_ok=True)
    result_dir = sample_dir / "results"

    upstream_workspace = sample_dir / "upstream_workspace"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for THEANINE runs.")
    workspace = ensure_upstream_workspace(
        theanine_dir=theanine_dir,
        workspace_dir=upstream_workspace,
        api_key=api_key,
    )
    upstream_data_dir = workspace["data_dir"]
    upstream_result_dir = workspace["result_dir"]

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
    trace["selected_history_indices"] = selected_indices
    trace["selected_session_ids"] = selected_session_ids
    trace["selected_session_dates"] = selected_session_dates
    trace["history_sessions_used"] = len(selected_indices)

    if dry_run:
        hypothesis = f"[dry-run] THEANINE would answer question {entry['question_id']}"
        trace["dry_run"] = True
        return hypothesis, trace

    random.seed(seed)
    with theanine_workspace_env(workspace["workspace_dir"], workspace["config_path"]):
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
    qa_llm = OpenAIAnswerClient(
        api_key=os.environ.get("OPENAI_API_KEY") or "",
        model=llm_model,
        temperature=0.0,
        max_tokens=256,
    )
    hypothesis = qa_llm.chat(build_unified_qa_messages(query, evidence_rows)).strip()
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

    dataset = load_dataset(args.longmemeval_file)
    dataset = dataset[args.offset :]
    if args.limit > 0:
        dataset = dataset[: args.limit]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl.exists():
        args.out_jsonl.unlink()
    if args.trace_jsonl and args.trace_jsonl.exists():
        args.trace_jsonl.unlink()
    audit_query_path, audit_write_path = derive_audit_paths(args.trace_jsonl)
    cf_run_path, cf_query_path = derive_cf_paths(args.trace_jsonl)
    for path in (audit_query_path, audit_write_path, cf_run_path, cf_query_path):
        if path is not None and path.exists():
            path.unlink()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

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
            write_records, query_record = build_theanine_audit_records(entry, trace, hypothesis)
            append_audit_jsonl(audit_query_path, query_record)
            for write_record in write_records:
                append_audit_jsonl(audit_write_path, write_record)
            if args.enable_cf_wrapper and not args.dry_run:
                cf_events = collect_theanine_write_events(
                    entry,
                    args.history_sessions,
                    args.preserve_session_order,
                )
                cf_hypothesis, cf_trace = run_theanine_for_entry(
                    entry=entry,
                    theanine_dir=args.theanine_dir,
                    runtime_dir=args.runtime_dir / "cf_baseline",
                    llm_model=args.llm_model,
                    temperature=args.temperature,
                    history_sessions=args.history_sessions,
                    preserve_session_order=args.preserve_session_order,
                    omit_question_date=args.omit_question_date,
                    dry_run=False,
                    verbose_upstream=args.verbose_upstream,
                    seed=args.seed + idx,
                    events=cf_events,
                )
                cf_write_records, cf_query_record = build_theanine_replay_audit(entry, cf_events, cf_trace, cf_hypothesis)
                specs = build_cf_specs(
                    question_type=entry.get("question_type", "unknown"),
                    query_record=cf_query_record,
                    write_records=cf_write_records,
                    answer_session_ids=entry.get("answer_session_ids", []),
                    max_writes=args.cf_max_writes,
                    scope=args.cf_target_scope,
                )
                cf_results = []
                for spec in specs:
                    mutated_events, target_timestamp = apply_theanine_cf_spec(cf_events, spec)
                    outcome_hypothesis, outcome_trace = run_theanine_for_entry(
                        entry=entry,
                        theanine_dir=args.theanine_dir,
                        runtime_dir=args.runtime_dir / "cf_runs",
                        llm_model=args.llm_model,
                        temperature=args.temperature,
                        history_sessions=args.history_sessions,
                        preserve_session_order=args.preserve_session_order,
                        omit_question_date=args.omit_question_date,
                        dry_run=False,
                        verbose_upstream=args.verbose_upstream,
                        seed=args.seed + idx,
                        events=mutated_events,
                    )
                    _, outcome_query_record = build_theanine_replay_audit(entry, mutated_events, outcome_trace, outcome_hypothesis)
                    cf_results.append(
                        {
                            "spec": spec,
                            "cf_answer": outcome_hypothesis,
                            "cf_retrieved_write_ids": outcome_query_record.get("retrieved_write_ids", []),
                            "cf_prompt_write_ids": outcome_query_record.get("prompt_write_ids", []),
                            "target_timestamp": target_timestamp,
                        }
                    )
                run_records, query_summary = summarize_replay_cf(
                    agent="theanine",
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
