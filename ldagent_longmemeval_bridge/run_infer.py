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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longmemeval_unified_answer import EvidenceRow, build_unified_qa_messages


@dataclass
class RetrievalSnapshot:
    context_memories: List[Dict]
    related_memories: List[Dict]


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


def get_ordered_sessions(entry: Dict, preserve_order: bool) -> List[Tuple[str, List[Dict]]]:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    pairs = [(d, s) for d, s in zip(dates, sessions)]

    if not preserve_order:
        indexed = []
        for idx, (date_raw, turns) in enumerate(pairs):
            dt = parse_longmemeval_datetime(date_raw)
            indexed.append((idx, dt, date_raw, turns))
        indexed.sort(
            key=lambda item: (
                1 if item[1] is None else 0,
                item[1] if item[1] is not None else datetime.max,
                item[0],
            )
        )
        return [(date_raw, turns) for _idx, _dt, date_raw, turns in indexed]

    return pairs


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

    def employ(self, system_prompt: str, user_prompt: str, name: str = "default") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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

    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
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
