#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

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


@contextmanager
def maybe_silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        yield


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


def get_ordered_session_entries(
    entry: Dict,
    preserve_order: bool,
) -> List[Tuple[int, str, str, List[Dict]]]:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])
    pairs = [
        (
            idx + 1,
            str(session_ids[idx]) if idx < len(session_ids) and session_ids[idx] else f"s{idx + 1}",
            date_raw,
            turns,
        )
        for idx, (date_raw, turns) in enumerate(zip(dates, sessions))
    ]

    if preserve_order:
        return pairs

    indexed = []
    for session_order, session_id, date_raw, turns in pairs:
        dt = parse_longmemeval_datetime(date_raw)
        indexed.append((session_order, dt, session_id, date_raw, turns))
    indexed.sort(
        key=lambda x: (
            1 if x[1] is None else 0,
            x[1] if x[1] is not None else datetime.max,
            x[0],
        )
    )
    return [(session_order, session_id, date_raw, turns) for session_order, _dt, session_id, date_raw, turns in indexed]


def get_ordered_sessions(entry: Dict, preserve_order: bool) -> List[Tuple[str, List[Dict]]]:
    return [(date_raw, turns) for _order, _session_id, date_raw, turns in get_ordered_session_entries(entry, preserve_order)]


def iter_qa_pairs(turns: List[Dict]) -> Iterable[Tuple[str, str]]:
    if not turns:
        return

    if all(isinstance(item, str) for item in turns):
        for item in turns:
            text = item.strip()
            if text:
                yield text, ""
        return

    pending_user: Optional[str] = None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "").strip().lower()
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


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def lexical_retrieve(question: str, items: List[str], top_k: int) -> List[str]:
    if not items or top_k <= 0:
        return []
    q_tokens = set(simple_tokenize(question))
    scored: List[Tuple[float, str]] = []
    for item in items:
        tok = set(simple_tokenize(item))
        overlap = len(q_tokens & tok)
        score = overlap / max(1, len(q_tokens)) if tok else 0.0
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item for score, item in scored[:top_k] if score > 0]
    if selected:
        return selected
    return items[:top_k]


@dataclass
class AnnaRetrievalResult:
    need_history: bool
    retrieved_text: str
    fallback_memories: List[str]


@dataclass
class MemoryUnit:
    text: str
    session_id: str
    timestamp: str
    turn_span: List[object]
    session_order: int


@dataclass
class MemoryViews:
    long_term_conversations: List[Dict]
    long_term_lines: List[str]
    long_term_units: List[MemoryUnit]
    short_term_lines: List[str]
    short_term_units: List[MemoryUnit]
    all_lines: List[str]
    all_units: List[MemoryUnit]
    n_pairs: int
    n_sessions: int
    n_long_term_sessions: int
    n_short_term_sessions: int
    used_long_term_fallback: bool


@dataclass
class AnnaTertiaryContext:
    profile: Dict
    report_payload: Dict
    status: str
    situation: str
    style: List[str]
    complaint_chain: List[Dict]
    scales: Dict


class OpenAITextClient:
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


class AnnaRetriever:
    def __init__(
        self,
        anna_agent_dir: Path,
        api_key: str,
        model: str,
        base_url: Optional[str],
    ) -> None:
        src_dir = anna_agent_dir / "src"
        if not src_dir.exists():
            raise FileNotFoundError(f"AnnaAgent src dir not found: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        os.environ.setdefault("ANNA_ENGINE_API_KEY", api_key)
        os.environ.setdefault("ANNA_ENGINE_COUNSELOR_API_KEY", api_key)
        os.environ.setdefault("ANNA_ENGINE_COMPLAINT_API_KEY", api_key)
        os.environ.setdefault("ANNA_ENGINE_EMOTION_API_KEY", api_key)
        os.environ.setdefault("ANNA_ENGINE_MODEL_NAME", model)
        os.environ.setdefault("ANNA_ENGINE_COUNSELOR_MODEL_NAME", model)
        os.environ.setdefault("ANNA_ENGINE_COMPLAINT_MODEL_NAME", model)
        os.environ.setdefault("ANNA_ENGINE_EMOTION_MODEL_NAME", model)
        if base_url:
            os.environ.setdefault("ANNA_ENGINE_BASE_URL", base_url)
            os.environ.setdefault("ANNA_ENGINE_COUNSELOR_BASE_URL", base_url)
            os.environ.setdefault("ANNA_ENGINE_COMPLAINT_BASE_URL", base_url)
            os.environ.setdefault("ANNA_ENGINE_EMOTION_BASE_URL", base_url)

        with maybe_silence_stdout(True):
            from anna_agent.backbone import configure
            from anna_agent.anna_agent_template import prompt_template
            from anna_agent.complaint_chain import gen_complaint_chain
            from anna_agent.event_trigger import situationalising_events
            from anna_agent.fill_scales import fill_scales, fill_scales_previous
            from anna_agent.short_term_memory import summarize_scale_changes
            from anna_agent.style_analyzer import analyze_style
            from anna_agent.querier import is_need, query

            configure(workspace=anna_agent_dir)

        self._is_need = is_need
        self._query = query
        self._prompt_template = prompt_template
        self._fill_scales = fill_scales
        self._fill_scales_previous = fill_scales_previous
        self._summarize_scale_changes = summarize_scale_changes
        self._situationalising_events = situationalising_events
        self._analyze_style = analyze_style
        self._gen_complaint_chain = gen_complaint_chain

    @staticmethod
    def _default_scale(size: int) -> List[str]:
        # Keep neutral distribution to avoid degenerate scale-change summaries.
        pattern = ["B", "C", "B", "A"]
        out = [pattern[i % len(pattern)] for i in range(size)]
        return out

    def _build_profile(self, lines: List[str]) -> Dict:
        joined = " ".join(lines).lower()
        age = 35
        age_match = re.search(r"\b([1-8][0-9])\b", joined)
        if age_match:
            age = int(age_match.group(1))
        gender = "Female"
        if any(token in joined for token in [" he ", " him ", " his ", " husband ", " boyfriend "]):
            gender = "Male"
        occupation = "Office worker"
        if "student" in joined:
            occupation = "Student"
        if "teacher" in joined:
            occupation = "Teacher"
        marital = "Married" if any(token in joined for token in ["wife", "husband", "married"]) else "Single"
        symptoms = "anxiety, stress, and sleep disturbance"
        return {
            "age": str(age),
            "gender": gender,
            "occupation": occupation,
            "martial_status": marital,
            "symptoms": symptoms,
            "symptom": symptoms,
        }

    def _build_tertiary_context(
        self,
        question: str,
        question_type: str,
        question_date: str,
        previous_conversations: List[Dict],
        long_term_lines: List[str],
        short_term_lines: List[str],
    ) -> AnnaTertiaryContext:
        profile = self._build_profile(long_term_lines or short_term_lines)
        base_report = {
            "question_type": question_type,
            "question_date": question_date,
            "question": question,
            "long_term_summary": "\n".join(long_term_lines[-50:]),
            "short_term_summary": "\n".join(short_term_lines[-30:]),
        }

        try:
            style = list(self._analyze_style(profile, previous_conversations))
        except Exception:
            style = ["guarded", "hesitant"]

        try:
            p_bdi, p_ghq, p_sass = self._fill_scales_previous(profile, base_report)
        except Exception:
            p_bdi = self._default_scale(21)
            p_ghq = self._default_scale(28)
            p_sass = self._default_scale(21)

        try:
            situation = str(self._situationalising_events(profile))
        except Exception:
            situation = "最近在工作和生活中感到持续压力。"

        seeker_utterances = [x.get("content", "") for x in previous_conversations if x.get("role") == "Seeker" and x.get("content")]
        if seeker_utterances:
            sample_size = min(3, len(seeker_utterances))
            statement = random.sample(seeker_utterances, k=sample_size)
        else:
            statement = ["最近工作压力很大，睡眠不太好。"]

        prompt = self._prompt_template.format(
            gender=profile["gender"],
            age=profile["age"],
            occupation=profile["occupation"],
            marriage=profile["martial_status"],
            situation=situation,
            status="",
            statement=statement,
            style=style,
        )

        try:
            bdi, ghq, sass = self._fill_scales(prompt)
        except Exception:
            bdi = list(p_bdi)
            ghq = list(p_ghq)
            sass = list(p_sass)

        scales = {
            "p_bdi": p_bdi,
            "p_ghq": p_ghq,
            "p_sass": p_sass,
            "bdi": bdi,
            "ghq": ghq,
            "sass": sass,
        }

        try:
            status = str(self._summarize_scale_changes(scales))
        except Exception:
            status = "近期情绪与压力指标波动，整体处于轻到中度困扰状态。"

        try:
            complaint_chain = list(self._gen_complaint_chain(profile))
        except Exception:
            complaint_chain = []

        report_payload = {
            "profile": profile,
            "status": status,
            "situation": situation,
            "style": style,
            "complaint_chain": complaint_chain,
            "scales": scales,
            "question_type": question_type,
            "question_date": question_date,
            "long_term_summary": base_report["long_term_summary"],
            "short_term_summary": base_report["short_term_summary"],
        }
        return AnnaTertiaryContext(
            profile=profile,
            report_payload=report_payload,
            status=status,
            situation=situation,
            style=style,
            complaint_chain=complaint_chain,
            scales=scales,
        )

    def retrieve(
        self,
        question: str,
        previous_conversations: List[Dict],
        report: Dict,
        use_need_check: bool,
        fallback_memories: List[str],
        question_type: str,
        question_date: str,
        long_term_lines: List[str],
        short_term_lines: List[str],
        enable_full_tertiary_init: bool,
    ) -> AnnaRetrievalResult:
        need_history = True
        retrieved_text = ""
        report_payload = dict(report)

        if enable_full_tertiary_init:
            tertiary = self._build_tertiary_context(
                question=question,
                question_type=question_type,
                question_date=question_date,
                previous_conversations=previous_conversations,
                long_term_lines=long_term_lines,
                short_term_lines=short_term_lines,
            )
            report_payload = tertiary.report_payload

        if use_need_check:
            try:
                need_history = bool(self._is_need(question))
            except Exception:
                need_history = True

        if need_history:
            try:
                retrieved_text = str(self._query(question, previous_conversations, report_payload)).strip()
            except Exception:
                retrieved_text = ""

        return AnnaRetrievalResult(
            need_history=need_history,
            retrieved_text=retrieved_text,
            fallback_memories=fallback_memories,
        )


def build_memory_views(
    entry: Dict,
    preserve_order: bool,
    short_term_sessions: int,
    strict_tertiary_split: bool,
) -> MemoryViews:
    session_conversations: List[List[Dict]] = []
    session_lines: List[List[str]] = []
    session_units: List[List[MemoryUnit]] = []
    pair_count = 0

    for session_order, session_id, date_raw, turns in get_ordered_session_entries(entry, preserve_order):
        current_session_conversations: List[Dict] = []
        current_session_lines: List[str] = []
        current_session_units: List[MemoryUnit] = []
        line_idx = 0
        for user_turn, assistant_turn in iter_qa_pairs(turns):
            if user_turn.strip():
                line_idx += 1
                current_session_conversations.append(
                    {
                        "role": "Seeker",
                        "content": user_turn.strip(),
                        "session_date": date_raw,
                    }
                )
                current_session_lines.append(f"Seeker: {user_turn.strip()}")
                current_session_units.append(
                    MemoryUnit(
                        text=f"Seeker: {user_turn.strip()}",
                        session_id=session_id,
                        timestamp=date_raw,
                        turn_span=[line_idx],
                        session_order=session_order,
                    )
                )
            if assistant_turn.strip():
                line_idx += 1
                current_session_conversations.append(
                    {
                        "role": "Counselor",
                        "content": assistant_turn.strip(),
                        "session_date": date_raw,
                    }
                )
                current_session_lines.append(f"Counselor: {assistant_turn.strip()}")
                current_session_units.append(
                    MemoryUnit(
                        text=f"Counselor: {assistant_turn.strip()}",
                        session_id=session_id,
                        timestamp=date_raw,
                        turn_span=[line_idx],
                        session_order=session_order,
                    )
                )
            pair_count += 1

        if current_session_conversations:
            session_conversations.append(current_session_conversations)
            session_lines.append(current_session_lines)
            session_units.append(current_session_units)

    n_sessions = len(session_conversations)
    bounded_short_sessions = min(max(short_term_sessions, 0), n_sessions)
    split_idx = n_sessions - bounded_short_sessions

    long_term_session_groups = session_conversations[:split_idx]
    long_term_line_groups = session_lines[:split_idx]
    long_term_unit_groups = session_units[:split_idx]
    short_term_line_groups = session_lines[split_idx:]
    short_term_unit_groups = session_units[split_idx:]
    used_long_term_fallback = False

    # For sparse samples, keep the retriever usable instead of passing an empty long-term memory.
    if not long_term_session_groups and not strict_tertiary_split:
        long_term_session_groups = session_conversations
        long_term_line_groups = session_lines
        long_term_unit_groups = session_units
        used_long_term_fallback = True

    long_term_conversations = [turn for group in long_term_session_groups for turn in group]
    long_term_lines = [line for group in long_term_line_groups for line in group]
    long_term_units = [unit for group in long_term_unit_groups for unit in group]
    short_term_lines = [line for group in short_term_line_groups for line in group]
    short_term_units = [unit for group in short_term_unit_groups for unit in group]
    all_lines = [line for group in session_lines for line in group]
    all_units = [unit for group in session_units for unit in group]

    return MemoryViews(
        long_term_conversations=long_term_conversations,
        long_term_lines=long_term_lines,
        long_term_units=long_term_units,
        short_term_lines=short_term_lines,
        short_term_units=short_term_units,
        all_lines=all_lines,
        all_units=all_units,
        n_pairs=pair_count,
        n_sessions=n_sessions,
        n_long_term_sessions=len(long_term_session_groups),
        n_short_term_sessions=len(short_term_line_groups),
        used_long_term_fallback=used_long_term_fallback,
    )


def collect_anna_write_events(entry: Dict, preserve_order: bool) -> List[MemoryUnit]:
    return build_memory_views(
        entry=entry,
        preserve_order=preserve_order,
        short_term_sessions=1,
        strict_tertiary_split=False,
    ).all_units


def sort_anna_units(units: Iterable[MemoryUnit]) -> List[MemoryUnit]:
    def sort_key(unit: MemoryUnit):
        turn = int(unit.turn_span[0]) if unit.turn_span else 0
        return (unit.session_order, turn)

    return sorted(list(units), key=sort_key)


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def anna_unit_match_fragments(unit: MemoryUnit) -> List[str]:
    text = unit.text.split(":", 1)[1].strip() if ":" in unit.text else unit.text.strip()
    fragments = {normalize_match_text(text)}
    for fragment in re.split(r"[。.!?;；\n]+", text):
        norm = normalize_match_text(fragment)
        if len(norm) >= 24:
            fragments.add(norm)
    return [fragment for fragment in fragments if fragment]


def map_anna_retrieved_units(retrieved_text: str, units: Iterable[MemoryUnit]) -> List[MemoryUnit]:
    norm_blob = normalize_match_text(retrieved_text)
    if not norm_blob:
        return []
    matched: List[MemoryUnit] = []
    seen = set()
    for unit in units:
        for fragment in anna_unit_match_fragments(unit):
            if fragment and fragment in norm_blob:
                key = (unit.session_id, tuple(unit.turn_span), unit.timestamp, unit.text)
                if key not in seen:
                    seen.add(key)
                    matched.append(unit)
                break
    return matched


def apply_anna_cf_spec(qid: str, units: Iterable[MemoryUnit], spec) -> Tuple[List[MemoryUnit], Optional[str]]:
    mutated: List[MemoryUnit] = []
    target_timestamp: Optional[str] = None
    for unit in units:
        write_id = make_write_id(
            agent="anna",
            question_id=qid,
            write_type="anna_memory_unit",
            content=unit.text,
            session_id=unit.session_id,
            turn_span=unit.turn_span,
            timestamp=unit.timestamp,
        )
        if write_id != spec.target_write_id:
            mutated.append(unit)
            continue
        if spec.cf_type == "rollback":
            target_timestamp = unit.timestamp
            continue
        mutated.append(
            MemoryUnit(
                text=unit.text,
                session_id=unit.session_id,
                timestamp=spec.new_timestamp or unit.timestamp,
                turn_span=list(unit.turn_span),
                session_order=unit.session_order,
            )
        )
        target_timestamp = spec.new_timestamp or unit.timestamp
    return sort_anna_units(mutated), target_timestamp


def build_memory_views_from_units(
    units: Iterable[MemoryUnit],
    short_term_sessions: int,
    strict_tertiary_split: bool,
) -> MemoryViews:
    grouped: Dict[str, List[MemoryUnit]] = {}
    for unit in sort_anna_units(units):
        grouped.setdefault(unit.session_id, []).append(unit)

    session_ids = sorted(grouped.keys(), key=lambda sid: min(unit.session_order for unit in grouped[sid]))
    session_lines = [[unit.text for unit in grouped[sid]] for sid in session_ids]
    session_units = [grouped[sid] for sid in session_ids]
    session_conversations: List[List[Dict]] = []
    pair_count = 0
    for sid in session_ids:
        convo: List[Dict] = []
        for unit in grouped[sid]:
            role = "Seeker" if unit.text.startswith("Seeker:") else "Counselor"
            content = unit.text.split(":", 1)[1].strip() if ":" in unit.text else unit.text
            convo.append({"role": role, "content": content, "session_date": unit.timestamp})
            if role == "Counselor":
                pair_count += 1
        session_conversations.append(convo)

    n_sessions = len(session_conversations)
    bounded_short_sessions = min(max(short_term_sessions, 0), n_sessions)
    split_idx = n_sessions - bounded_short_sessions
    long_term_session_groups = session_conversations[:split_idx]
    long_term_line_groups = session_lines[:split_idx]
    long_term_unit_groups = session_units[:split_idx]
    short_term_line_groups = session_lines[split_idx:]
    short_term_unit_groups = session_units[split_idx:]
    used_long_term_fallback = False
    if not long_term_session_groups and not strict_tertiary_split:
        long_term_session_groups = session_conversations
        long_term_line_groups = session_lines
        long_term_unit_groups = session_units
        used_long_term_fallback = True

    return MemoryViews(
        long_term_conversations=[turn for group in long_term_session_groups for turn in group],
        long_term_lines=[line for group in long_term_line_groups for line in group],
        long_term_units=[unit for group in long_term_unit_groups for unit in group],
        short_term_lines=[line for group in short_term_line_groups for line in group],
        short_term_units=[unit for group in short_term_unit_groups for unit in group],
        all_lines=[line for group in session_lines for line in group],
        all_units=[unit for group in session_units for unit in group],
        n_pairs=pair_count,
        n_sessions=n_sessions,
        n_long_term_sessions=len(long_term_session_groups),
        n_short_term_sessions=len(short_term_line_groups),
        used_long_term_fallback=used_long_term_fallback,
    )


def make_report(
    entry: Dict,
    long_term_lines: List[str],
    short_term_lines: List[str],
    max_lines: int,
) -> Dict:
    long_term_clipped = long_term_lines[-max_lines:] if max_lines > 0 else long_term_lines
    short_term_clipped = short_term_lines[-max_lines:] if max_lines > 0 else short_term_lines
    long_term_text = "\n".join(long_term_clipped)
    short_term_text = "\n".join(short_term_clipped)
    return {
        "案例标题": "LongMemEval 多轮会话记忆检索",
        "案例类别": [str(entry.get("question_type", "unknown"))],
        "咨询经过": [long_term_text] if long_term_text else [],
        "当前短期状态": [short_term_text] if short_term_text else [],
        "问题日期": str(entry.get("question_date", "")),
        # Compatibility aliases for bridge debugging / backward traces.
        "question_type": entry.get("question_type", "unknown"),
        "question_date": entry.get("question_date", ""),
        "long_term_summary": long_term_text,
        "short_term_summary": short_term_text,
    }


def answer_with_anna_style(
    llm: OpenAITextClient,
    question: str,
    query_with_date: str,
    retrieved_text: str,
    fallback_memories: List[str],
    short_term_context: List[str],
    real_time_context: str,
    dry_run: bool,
) -> str:
    if dry_run:
        return "DRY_RUN_PLACEHOLDER"

    del question
    evidence_rows: List[EvidenceRow] = []
    if retrieved_text:
        evidence_rows.append(EvidenceRow(text=retrieved_text, source="anna_long_term"))
    evidence_rows.extend(EvidenceRow(text=item, source="anna_fallback") for item in fallback_memories)
    evidence_rows.extend(EvidenceRow(text=item, source="anna_short_term") for item in short_term_context)
    if real_time_context:
        evidence_rows.append(EvidenceRow(text=real_time_context, source="anna_real_time"))
    return llm.chat(build_unified_qa_messages(query_with_date, evidence_rows)).strip()


def run_anna_replay(
    *,
    entry: Dict,
    units: List[MemoryUnit],
    args: argparse.Namespace,
    retriever: "AnnaRetriever",
    llm: "OpenAITextClient",
) -> Dict:
    qid = entry["question_id"]
    qtype = entry.get("question_type", "unknown")
    memory_views = build_memory_views_from_units(
        units,
        short_term_sessions=args.short_term_sessions,
        strict_tertiary_split=args.strict_tertiary_split,
    )
    question = entry.get("question", "").strip()
    query_with_date = question
    if not args.omit_question_date and entry.get("question_date"):
        query_with_date = f"Current date: {entry['question_date']}\n\n{question}"

    report = make_report(
        entry,
        memory_views.long_term_lines,
        memory_views.short_term_lines,
        args.report_summary_lines,
    )
    fallback_source = memory_views.long_term_lines or memory_views.short_term_lines
    fallback_memories = lexical_retrieve(
        question=question,
        items=fallback_source,
        top_k=args.fallback_top_k,
    )
    retrieval = retriever.retrieve(
        question=question,
        previous_conversations=memory_views.long_term_conversations,
        report=report,
        use_need_check=not args.disable_need_check,
        fallback_memories=fallback_memories,
        question_type=qtype,
        question_date=str(entry.get("question_date", "")),
        long_term_lines=memory_views.long_term_lines,
        short_term_lines=memory_views.short_term_lines,
        enable_full_tertiary_init=args.enable_full_tertiary_init,
    )
    short_term_source = memory_views.short_term_lines or memory_views.all_lines
    short_term = short_term_source[-args.short_term_window :] if args.short_term_window > 0 else short_term_source
    real_time_context = f"Counselor: {question}"
    hypothesis = answer_with_anna_style(
        llm=llm,
        question=question,
        query_with_date=query_with_date,
        retrieved_text=retrieval.retrieved_text,
        fallback_memories=retrieval.fallback_memories,
        short_term_context=short_term,
        real_time_context=real_time_context,
        dry_run=args.dry_run,
    )

    write_records: List[Dict] = []
    candidate_write_ids: List[str] = []
    retrieved_write_ids: List[str] = []
    selected_write_ids: List[str] = []
    prompt_write_ids: List[str] = []
    retrieved_items: List[Dict] = []
    prompt_items: List[Dict] = []
    bridge_items: List[Dict] = []
    unit_index: Dict[str, List[Tuple[MemoryUnit, str]]] = {}
    write_order = 0
    for unit in memory_views.all_units:
        write_order += 1
        write_record, _ = build_anna_write_record(
            qid=qid,
            unit=unit,
            write_type="anna_memory_unit",
            stage="write_ingress",
            write_order=write_order,
        )
        write_records.append(write_record)
        candidate_write_ids.append(write_record["write_id"])
        unit_index.setdefault(unit.text, []).append((unit, "anna_memory_unit"))

    seen_retrieved = set()
    matched_long_term_ids: List[str] = []
    matched_long_term_sessions: List[str] = []
    matched_long_term_timestamps: List[str] = []
    for unit in map_anna_retrieved_units(retrieval.retrieved_text, memory_views.long_term_units):
        write_id = make_write_id(
            agent="anna",
            question_id=qid,
            write_type="anna_memory_unit",
            content=unit.text,
            session_id=unit.session_id,
            turn_span=unit.turn_span,
            timestamp=unit.timestamp,
        )
        if write_id in seen_retrieved:
            continue
        seen_retrieved.add(write_id)
        matched_long_term_ids.append(write_id)
        if unit.session_id:
            matched_long_term_sessions.append(unit.session_id)
        if unit.timestamp is not None:
            matched_long_term_timestamps.append(str(unit.timestamp))

    if matched_long_term_ids:
        item_record = build_item_record(
            write_id=matched_long_term_ids[0] if len(matched_long_term_ids) == 1 else None,
            source_write_ids=matched_long_term_ids,
            source_session_ids=matched_long_term_sessions,
            event_timestamps=matched_long_term_timestamps,
            memory_timestamps=matched_long_term_timestamps,
            stage="anna_long_term_retrieval",
            rank=len(prompt_items) + 1,
            score=None,
            timestamp=matched_long_term_timestamps[0] if matched_long_term_timestamps else None,
            write_type="anna_memory_unit",
            source_form="anna_retrieved_text",
            audit_eligible=True,
            text=retrieval.retrieved_text,
            source="anna_long_term_retrieval",
        )
        retrieved_write_ids.extend(matched_long_term_ids)
        prompt_write_ids.extend(matched_long_term_ids)
        selected_write_ids.extend(matched_long_term_ids)
        retrieved_items.append(dict(item_record))
        prompt_items.append(dict(item_record))

    seen_prompt = set(seen_retrieved)
    for line in short_term:
        for unit, write_type in unit_index.get(line, []):
            write_id = make_write_id(
                agent="anna",
                question_id=qid,
                write_type=write_type,
                content=unit.text,
                session_id=unit.session_id,
                turn_span=unit.turn_span,
                timestamp=unit.timestamp,
            )
            if write_id in seen_prompt:
                continue
            seen_prompt.add(write_id)
            prompt_write_ids.append(write_id)
            selected_write_ids.append(write_id)
            prompt_items.append(
                build_item_record(
                    write_id=write_id,
                    source_write_ids=[write_id],
                    source_session_ids=[unit.session_id] if unit.session_id else [],
                    event_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                    memory_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                    stage="short_term_prompt",
                    rank=len(prompt_items) + 1,
                    score=None,
                    timestamp=unit.timestamp,
                    write_type=write_type,
                    source_form="anna_short_term",
                    audit_eligible=True,
                    text=line,
                    source="anna_short_term",
                )
            )

    for item in retrieval.fallback_memories:
        matched = unit_index.get(item, [])
        if matched:
            for unit, write_type in matched:
                write_id = make_write_id(
                    agent="anna",
                    question_id=qid,
                    write_type=write_type,
                    content=unit.text,
                    session_id=unit.session_id,
                    turn_span=unit.turn_span,
                    timestamp=unit.timestamp,
                )
                if write_id in seen_prompt:
                    continue
                seen_prompt.add(write_id)
                prompt_write_ids.append(write_id)
                selected_write_ids.append(write_id)
                prompt_items.append(
                    build_item_record(
                        write_id=write_id,
                        source_write_ids=[write_id],
                        source_session_ids=[unit.session_id] if unit.session_id else [],
                        event_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                        memory_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                        stage="fallback_prompt",
                        rank=len(prompt_items) + 1,
                        score=None,
                        timestamp=unit.timestamp,
                        write_type=write_type,
                        source_form="anna_fallback",
                        audit_eligible=True,
                        text=item,
                        source="anna_fallback",
                    )
                )
        else:
            bridge_items.append(
                {
                    "text": item,
                    "source": "anna_fallback",
                    "source_form": "anna_fallback",
                    "audit_eligible": False,
                }
            )
    if retrieval.retrieved_text and not matched_long_term_ids:
        bridge_items.append(
            {
                "text": retrieval.retrieved_text,
                "source": "anna_native_retrieval_blob",
                "source_form": "anna_retrieved_text",
                "audit_eligible": False,
            }
        )
    if real_time_context:
        bridge_items.append(
            {
                "text": real_time_context,
                "source": "anna_real_time",
                "source_form": "anna_real_time",
                "audit_eligible": False,
            }
        )

    query_record = build_query_record(
        agent="anna",
        question_id=qid,
        question_type=qtype,
        query_time=entry.get("question_date"),
        question_date_used=entry.get("question_date"),
        baseline_answer=hypothesis,
        candidate_write_ids=candidate_write_ids,
        retrieved_write_ids=retrieved_write_ids,
        selected_write_ids=selected_write_ids,
        prompt_write_ids=prompt_write_ids,
        retrieved_items=retrieved_items,
        prompt_items=prompt_items,
        bridge_items=bridge_items,
        extra={
            "query_used": query_with_date,
            "need_history": retrieval.need_history,
            "used_long_term_fallback": memory_views.used_long_term_fallback,
        },
    )
    trace_obj = {
        "question_id": qid,
        "question_type": qtype,
        "role_mapping": "Seeker<-user,Counselor<-assistant",
        "full_tertiary_init": args.enable_full_tertiary_init,
        "need_check_enabled": not args.disable_need_check,
        "n_pairs": memory_views.n_pairs,
        "n_sessions": memory_views.n_sessions,
        "n_long_term_sessions": memory_views.n_long_term_sessions,
        "n_short_term_sessions": memory_views.n_short_term_sessions,
        "used_long_term_fallback": memory_views.used_long_term_fallback,
        "need_history": retrieval.need_history,
        "anna_retrieved_text": retrieval.retrieved_text,
        "fallback_memories": retrieval.fallback_memories,
        "short_term_context": short_term,
        "real_time_context": real_time_context,
        "query_used": query_with_date,
    }
    return {
        "hypothesis": hypothesis,
        "trace": trace_obj,
        "query_record": query_record,
        "write_records": write_records,
        "units": sort_anna_units(units),
    }


def build_anna_write_record(
    qid: str,
    unit: MemoryUnit,
    write_type: str,
    stage: str,
    write_order: int,
) -> Tuple[Dict, Dict]:
    write_id = make_write_id(
        agent="anna",
        question_id=qid,
        write_type=write_type,
        content=unit.text,
        session_id=unit.session_id,
        turn_span=unit.turn_span,
        timestamp=unit.timestamp,
    )
    write_record = build_write_record(
        agent="anna",
        question_id=qid,
        write_id=write_id,
        write_order=write_order,
        write_type=write_type,
        stage=stage,
        timestamp=unit.timestamp,
        session_id=unit.session_id,
        turn_span=unit.turn_span,
        content_text=unit.text,
        lineage_source_ids=[unit.session_id, f"{unit.session_id}:turn:{unit.turn_span[0]}"],
        audit_eligible=True,
        origin="native_memory",
    )
    item_record = {
        "write_id": write_id,
        "stage": stage,
        "rank": write_order,
        "score": None,
        "timestamp": str(unit.timestamp),
        "write_type": write_type,
        "audit_eligible": True,
    }
    return write_record, item_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnnaAgent-style memory retrieval on LongMemEval and export predictions JSONL."
    )
    parser.add_argument("--anna-agent-dir", type=Path, required=True, help="Path to AnnaAgent repository root.")
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--short-term-sessions",
        type=int,
        default=1,
        help="How many latest sessions are treated as short-term memory.",
    )
    parser.add_argument("--short-term-window", type=int, default=14)
    parser.add_argument("--fallback-top-k", type=int, default=6)
    parser.add_argument("--report-summary-lines", type=int, default=24)
    parser.add_argument(
        "--enable-full-tertiary-init",
        dest="enable_full_tertiary_init",
        action="store_true",
        help="Enable full Anna tertiary-memory initialization (style/scales/situation/status) before retrieval.",
    )
    parser.add_argument(
        "--disable-full-tertiary-init",
        dest="enable_full_tertiary_init",
        action="store_false",
        help="Disable full Anna tertiary-memory initialization (default).",
    )
    parser.set_defaults(enable_full_tertiary_init=False)
    parser.add_argument(
        "--strict-tertiary-split",
        action="store_true",
        help="Do not backfill long-term memory when short-term split consumes all sessions.",
    )
    parser.add_argument(
        "--disable-need-check",
        dest="disable_need_check",
        action="store_true",
        help="Disable Anna is_need gate and always run long-term query (default).",
    )
    parser.add_argument(
        "--enable-need-check",
        dest="disable_need_check",
        action="store_false",
        help="Enable Anna is_need gate before long-term query.",
    )
    parser.set_defaults(disable_need_check=True)
    parser.add_argument("--omit-question-date", action="store_true")
    parser.add_argument("--preserve-session-order", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-override", action="store_true")
    add_cf_args(parser)
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {path}, got {type(data)}")
    return data


def main() -> None:
    args = parse_args()

    if not args.anna_agent_dir.exists():
        raise FileNotFoundError(f"AnnaAgent repo not found: {args.anna_agent_dir}")

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

        llm = OpenAITextClient(
            api_key=api_key,
            model=args.llm_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        anna_retriever = AnnaRetriever(
            anna_agent_dir=args.anna_agent_dir,
            api_key=api_key,
            model=args.llm_model,
            base_url=args.openai_base_url,
        )
    else:
        llm = None
        anna_retriever = None

    success = 0
    failed = 0
    started = time.time()

    with args.out_jsonl.open("w", encoding="utf-8") as pred_file:
        trace_file = args.trace_jsonl.open("w", encoding="utf-8") if args.trace_jsonl else None
        try:
            pbar = tqdm(dataset, total=len(dataset), desc="AnnaAgent->LongMemEval", unit="q")
            for idx, entry in enumerate(pbar, start=1):
                qid = entry.get("question_id", f"idx_{idx}")
                qtype = entry.get("question_type", "unknown")
                try:
                    question = (entry.get("question") or "").strip()
                    query_with_date = question
                    if not args.omit_question_date and entry.get("question_date"):
                        query_with_date = f"Current date: {entry['question_date']}\n\n{question}"

                    memory_views = build_memory_views(
                        entry=entry,
                        preserve_order=args.preserve_session_order,
                        short_term_sessions=args.short_term_sessions,
                        strict_tertiary_split=args.strict_tertiary_split,
                    )
                    report = make_report(
                        entry=entry,
                        long_term_lines=memory_views.long_term_lines,
                        short_term_lines=memory_views.short_term_lines,
                        max_lines=args.report_summary_lines,
                    )

                    fallback_source = memory_views.long_term_lines
                    if not fallback_source:
                        fallback_source = memory_views.short_term_lines
                    memory_candidates = lexical_retrieve(
                        question=question,
                        items=fallback_source,
                        top_k=args.fallback_top_k,
                    )

                    if args.dry_run:
                        retrieval = AnnaRetrievalResult(
                            need_history=True,
                            retrieved_text="",
                            fallback_memories=memory_candidates,
                        )
                    else:
                        retrieval = anna_retriever.retrieve(
                            question=question,
                            previous_conversations=memory_views.long_term_conversations,
                            report=report,
                            use_need_check=not args.disable_need_check,
                            fallback_memories=memory_candidates,
                            question_type=qtype,
                            question_date=str(entry.get("question_date", "")),
                            long_term_lines=memory_views.long_term_lines,
                            short_term_lines=memory_views.short_term_lines,
                            enable_full_tertiary_init=args.enable_full_tertiary_init,
                        )

                    short_term_source = memory_views.short_term_lines
                    if not short_term_source:
                        short_term_source = memory_views.all_lines
                    short_term = (
                        short_term_source[-args.short_term_window :]
                        if args.short_term_window > 0
                        else short_term_source
                    )
                    real_time_context = f"Counselor: {question}"
                    hypothesis = answer_with_anna_style(
                        llm=llm,
                        question=question,
                        query_with_date=query_with_date,
                        retrieved_text=retrieval.retrieved_text,
                        fallback_memories=retrieval.fallback_memories,
                        short_term_context=short_term,
                        real_time_context=real_time_context,
                        dry_run=args.dry_run,
                    )

                    pred_obj = {"question_id": qid, "hypothesis": hypothesis}
                    pred_file.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_file.flush()

                    if trace_file:
                        trace_obj = {
                            "question_id": qid,
                            "question_type": qtype,
                            "role_mapping": "Seeker<-user,Counselor<-assistant",
                            "full_tertiary_init": args.enable_full_tertiary_init,
                            "need_check_enabled": not args.disable_need_check,
                            "n_pairs": memory_views.n_pairs,
                            "n_sessions": memory_views.n_sessions,
                            "n_long_term_sessions": memory_views.n_long_term_sessions,
                            "n_short_term_sessions": memory_views.n_short_term_sessions,
                            "used_long_term_fallback": memory_views.used_long_term_fallback,
                            "need_history": retrieval.need_history,
                            "anna_retrieved_text": retrieval.retrieved_text,
                            "fallback_memories": retrieval.fallback_memories,
                            "short_term_context": short_term,
                            "real_time_context": real_time_context,
                            "query_used": query_with_date,
                        }
                        trace_file.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")
                        trace_file.flush()

                    candidate_write_ids: List[str] = []
                    retrieved_write_ids: List[str] = []
                    selected_write_ids: List[str] = []
                    prompt_write_ids: List[str] = []
                    retrieved_items: List[Dict] = []
                    prompt_items: List[Dict] = []
                    bridge_items: List[Dict] = []
                    write_records: List[Dict] = []
                    unit_index: Dict[str, List[Tuple[MemoryUnit, str]]] = {}
                    write_order = 0

                    for write_type, units in (
                        ("anna_long_term_line", memory_views.long_term_units),
                        ("anna_short_term_line", memory_views.short_term_units),
                    ):
                        stage = "long_term_memory" if "long_term" in write_type else "short_term_memory"
                        for unit in units:
                            write_order += 1
                            write_record, item_record = build_anna_write_record(
                                qid=qid,
                                unit=unit,
                                write_type=write_type,
                                stage=stage,
                                write_order=write_order,
                            )
                            write_records.append(write_record)
                            candidate_write_ids.append(write_record["write_id"])
                            unit_index.setdefault(unit.text, []).append((unit, write_type))

                    seen_prompt = set()
                    matched_long_term_ids: List[str] = []
                    matched_long_term_sessions: List[str] = []
                    matched_long_term_timestamps: List[str] = []
                    for unit in map_anna_retrieved_units(retrieval.retrieved_text, memory_views.long_term_units):
                        write_id = make_write_id(
                            agent="anna",
                            question_id=qid,
                            write_type="anna_memory_unit",
                            content=unit.text,
                            session_id=unit.session_id,
                            turn_span=unit.turn_span,
                            timestamp=unit.timestamp,
                        )
                        if write_id in seen_prompt:
                            continue
                        seen_prompt.add(write_id)
                        matched_long_term_ids.append(write_id)
                        if unit.session_id:
                            matched_long_term_sessions.append(unit.session_id)
                        if unit.timestamp is not None:
                            matched_long_term_timestamps.append(str(unit.timestamp))

                    if matched_long_term_ids:
                        item_record = build_item_record(
                            write_id=matched_long_term_ids[0] if len(matched_long_term_ids) == 1 else None,
                            source_write_ids=matched_long_term_ids,
                            source_session_ids=matched_long_term_sessions,
                            event_timestamps=matched_long_term_timestamps,
                            memory_timestamps=matched_long_term_timestamps,
                            stage="anna_long_term_retrieval",
                            rank=len(prompt_items) + 1,
                            score=None,
                            timestamp=matched_long_term_timestamps[0] if matched_long_term_timestamps else None,
                            write_type="anna_memory_unit",
                            source_form="anna_retrieved_text",
                            audit_eligible=True,
                            text=retrieval.retrieved_text,
                            source="anna_long_term_retrieval",
                        )
                        retrieved_write_ids.extend(matched_long_term_ids)
                        prompt_write_ids.extend(matched_long_term_ids)
                        selected_write_ids.extend(matched_long_term_ids)
                        retrieved_items.append(dict(item_record))
                        prompt_items.append(dict(item_record))

                    for line in short_term:
                        for unit, write_type in unit_index.get(line, []):
                            write_id = make_write_id(
                                agent="anna",
                                question_id=qid,
                                write_type=write_type,
                                content=unit.text,
                                session_id=unit.session_id,
                                turn_span=unit.turn_span,
                                timestamp=unit.timestamp,
                            )
                            if write_id in seen_prompt:
                                continue
                            seen_prompt.add(write_id)
                            prompt_write_ids.append(write_id)
                            selected_write_ids.append(write_id)
                            prompt_items.append(
                                build_item_record(
                                    write_id=write_id,
                                    source_write_ids=[write_id],
                                    source_session_ids=[unit.session_id] if unit.session_id else [],
                                    event_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                                    memory_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                                    stage="short_term_prompt",
                                    rank=len(prompt_items) + 1,
                                    score=None,
                                    timestamp=unit.timestamp,
                                    write_type=write_type,
                                    source_form="anna_short_term",
                                    audit_eligible=True,
                                    text=line,
                                    source="anna_short_term",
                                )
                            )

                    for item in retrieval.fallback_memories:
                        matched = unit_index.get(item, [])
                        if matched:
                            for unit, write_type in matched:
                                write_id = make_write_id(
                                    agent="anna",
                                    question_id=qid,
                                    write_type=write_type,
                                    content=unit.text,
                                    session_id=unit.session_id,
                                    turn_span=unit.turn_span,
                                    timestamp=unit.timestamp,
                                )
                                if write_id in seen_prompt:
                                    continue
                                seen_prompt.add(write_id)
                                prompt_write_ids.append(write_id)
                                selected_write_ids.append(write_id)
                                prompt_items.append(
                                    build_item_record(
                                        write_id=write_id,
                                        source_write_ids=[write_id],
                                        source_session_ids=[unit.session_id] if unit.session_id else [],
                                        event_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                                        memory_timestamps=[unit.timestamp] if unit.timestamp is not None else [],
                                        stage="fallback_prompt",
                                        rank=len(prompt_items) + 1,
                                        score=None,
                                        timestamp=unit.timestamp,
                                        write_type=write_type,
                                        source_form="anna_fallback",
                                        audit_eligible=True,
                                        text=item,
                                        source="anna_fallback",
                                    )
                                )
                        else:
                            bridge_items.append(
                                {
                                    "text": item,
                                    "source": "anna_fallback",
                                    "source_form": "anna_fallback",
                                    "audit_eligible": False,
                                }
                            )

                    if retrieval.retrieved_text and not matched_long_term_ids:
                        bridge_items.append(
                            {
                                "text": retrieval.retrieved_text,
                                "source": "anna_native_retrieval_blob",
                                "source_form": "anna_retrieved_text",
                                "audit_eligible": False,
                            }
                        )
                    if real_time_context:
                        bridge_items.append(
                            {
                                "text": real_time_context,
                                "source": "anna_real_time",
                                "source_form": "anna_real_time",
                                "audit_eligible": False,
                            }
                        )

                    query_record = build_query_record(
                        agent="anna",
                        question_id=qid,
                        question_type=qtype,
                        query_time=entry.get("question_date"),
                        question_date_used=entry.get("question_date"),
                        baseline_answer=hypothesis,
                        candidate_write_ids=candidate_write_ids,
                        retrieved_write_ids=retrieved_write_ids,
                        selected_write_ids=selected_write_ids,
                        prompt_write_ids=prompt_write_ids,
                        retrieved_items=retrieved_items,
                        prompt_items=prompt_items,
                        bridge_items=bridge_items,
                        extra={
                            "query_used": query_with_date,
                            "need_history": retrieval.need_history,
                            "used_long_term_fallback": memory_views.used_long_term_fallback,
                        },
                    )
                    append_audit_jsonl(audit_query_path, query_record)
                    for write_record in write_records:
                        append_audit_jsonl(audit_write_path, write_record)
                    if args.enable_cf_wrapper and not args.dry_run:
                        cf_units = collect_anna_write_events(entry, args.preserve_session_order)
                        cf_baseline = run_anna_replay(
                            entry=entry,
                            units=cf_units,
                            args=args,
                            retriever=anna_retriever,
                            llm=llm,
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
                            mutated_units, target_timestamp = apply_anna_cf_spec(
                                entry["question_id"],
                                cf_baseline["units"],
                                spec,
                            )
                            cf_outcome = run_anna_replay(
                                entry=entry,
                                units=mutated_units,
                                args=args,
                                retriever=anna_retriever,
                                llm=llm,
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
                            agent="anna",
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
                    elapsed = time.time() - started
                    pbar.set_postfix(ok=success, fail=failed, last=qid, elapsed_s=f"{elapsed:.1f}")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    pred_obj = {"question_id": qid, "hypothesis": f"ERROR: {exc}"}
                    pred_file.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_file.flush()
                    tqdm.write(f"FAIL qid={qid}: {exc}")
                    if args.fail_fast:
                        raise
        finally:
            if trace_file:
                trace_file.close()

    elapsed = time.time() - started
    print(
        f"Done. success={success} failed={failed} total={len(dataset)} "
        f"time={elapsed:.1f}s out={args.out_jsonl}"
    )
    if args.trace_jsonl:
        print(f"Trace saved to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
