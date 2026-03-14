from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional


UNIFIED_QA_SYSTEM_PROMPT = """You are answering a long-term memory benchmark question.

Use only the provided evidence.
Do not role-play. Do not continue the conversation. Do not use outside knowledge.

Rules:
1. Base the answer only on the provided evidence.
2. If the evidence includes timestamps, dates, or temporal order, use them explicitly.
3. If multiple evidence items conflict, prefer the evidence that is most directly relevant and temporally valid for the question.
4. If the evidence is insufficient, answer exactly: I don't know.
5. Return only the final concise answer, with no explanation."""


@dataclass
class EvidenceRow:
    text: str
    source: str = "memory"
    timestamp: Optional[str] = None


def _normalize_timestamp(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "unknown"
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "unknown"
    return "unknown"


def build_evidence_block(rows: Iterable[EvidenceRow]) -> str:
    normalized: List[str] = []
    for idx, row in enumerate(rows, start=1):
        text = " ".join((row.text or "").split()).strip()
        if not text:
            continue
        timestamp = _normalize_timestamp(row.timestamp)
        normalized.append(
            f"[Memory {idx} | source={row.source} | date={timestamp}]\n{text}"
        )
    if not normalized:
        return "[Memory 1 | source=none | date=unknown]\n(none)"
    return "\n\n".join(normalized)


def build_unified_qa_messages(query_with_date: str, rows: Iterable[EvidenceRow]):
    evidence_block = build_evidence_block(rows)
    user_prompt = (
        f"Question:\n{query_with_date}\n\n"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        "Return only the final concise answer."
    )
    return [
        {"role": "system", "content": UNIFIED_QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
