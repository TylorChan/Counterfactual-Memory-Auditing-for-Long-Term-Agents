#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai_prompt_cache import install_openai_prompt_cache

install_openai_prompt_cache("mem0_official")

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

AGENT = "mem0_official"
WRITE_TYPE = "mem0_official_conversation_pair"
PRIMARY_STAGE = "mem0_official_search_result"
PRIMARY_SOURCE_FORM = "mem0_official_memory"
DEFAULT_BENCHMARKS_REF = "f75666d33ef560f0f196746e0e16c515d17e6856"


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
        description=(
            "Run the official mem0ai/memory-benchmarks LongMemEval pipeline with "
            "write-time rollback CF auditing."
        )
    )
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument(
        "--memory-benchmarks-dir",
        type=Path,
        default=REPO_ROOT / "external" / "memory-benchmarks",
        help="Path to a mem0ai/memory-benchmarks checkout.",
    )
    parser.add_argument(
        "--auto-clone-memory-benchmarks",
        action="store_true",
        help="Clone the official benchmark repo if --memory-benchmarks-dir is missing.",
    )
    parser.add_argument("--memory-benchmarks-ref", default=DEFAULT_BENCHMARKS_REF)
    parser.add_argument("--project-name", default="mem0-official-cf")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--backend", choices=("cloud", "oss"), default="cloud")
    parser.add_argument("--mem0-host", default=None)
    parser.add_argument("--mem0-api-key", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--answerer-model", default="gpt-5")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--answer-cutoff", type=int, default=200)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--score-debug", action="store_true")
    parser.add_argument("--rpm", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="0 means all entries after offset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-add-retries", type=int, default=5)
    parser.add_argument("--add-event-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--event-poll-timeout", type=float, default=300.0)
    parser.add_argument("--fill-created-at-from-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--show-ingest-progress",
        action="store_true",
        help="Show a per-write progress bar during Mem0 ingestion. Useful for local smoke tests.",
    )
    parser.add_argument("--cleanup-users", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--cf-rule-mode",
        choices=("rollback-only", "all"),
        default="rollback-only",
        help="Official 94%% protocol defaults to rollback-only; 'all' also enables time-shift specs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    add_cf_args(parser)
    return parser.parse_args()


def ensure_memory_benchmarks_repo(path: Path, *, auto_clone: bool, ref: str) -> Path:
    lock_path = path.parent / ".memory-benchmarks.setup.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            if not auto_clone:
                raise RuntimeError(
                    f"Missing memory-benchmarks checkout at {path}. Run "
                    "`bash scripts/setup_mem0_official_benchmarks.sh` or pass "
                    "--auto-clone-memory-benchmarks."
                )
            subprocess.run(
                ["git", "clone", "https://github.com/mem0ai/memory-benchmarks.git", str(path)],
                check=True,
            )
        if ref:
            current = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
            if current != ref:
                subprocess.run(["git", "-C", str(path), "fetch", "--all", "--tags", "--depth", "1"], check=False)
                subprocess.run(["git", "-C", str(path), "checkout", ref], check=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if not (path / "benchmarks" / "longmemeval" / "run.py").exists():
        raise RuntimeError(f"Invalid memory-benchmarks checkout: {path}")
    return path.resolve()


def import_official_modules(benchmarks_dir: Path):
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    from benchmarks.common.llm_client import LLMClient  # type: ignore
    from benchmarks.common.mem0_client import Mem0Client  # type: ignore
    from benchmarks.longmemeval.prompts import get_answer_generation_prompt  # type: ignore
    from benchmarks.longmemeval.run import (  # type: ignore
        pair_turns,
        parse_longmemeval_date,
        parse_longmemeval_date_human,
        sort_sessions_chronologically,
    )

    return {
        "LLMClient": LLMClient,
        "Mem0Client": Mem0Client,
        "get_answer_generation_prompt": get_answer_generation_prompt,
        "pair_turns": pair_turns,
        "parse_longmemeval_date": parse_longmemeval_date,
        "parse_longmemeval_date_human": parse_longmemeval_date_human,
        "sort_sessions_chronologically": sort_sessions_chronologically,
    }


class CurrentMem0CloudClient:
    """Async Mem0 Cloud client matching the current mem0ai SDK endpoints.

    The pinned memory-benchmarks repo still posts adds to /v3/memories/.
    mem0ai==2.x uses /v3/memories/add/, so this adapter keeps the official
    benchmark prompt/replay logic while using the current cloud API shape.
    """

    def __init__(
        self,
        *,
        host: Optional[str],
        api_key: str,
        max_retries: int,
        event_retries: int,
        retry_delay: float,
        timeout: float,
        rpm: int,
        event_poll_timeout: float = 300.0,
        event_poll_interval: float = 0.5,
    ) -> None:
        import aiohttp
        from aiolimiter import AsyncLimiter

        self.host = (host or "https://api.mem0.ai").rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.event_retries = max(1, event_retries)
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.limiter = AsyncLimiter(rpm, 60)
        self.event_poll_timeout = event_poll_timeout
        self.event_poll_interval = event_poll_interval
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers, timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "CurrentMem0CloudClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @staticmethod
    def _normalise_add_response(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                return data
            if isinstance(data.get("memories"), list):
                return {"results": data["memories"], "raw": data}
            if "id" in data or "event" in data or "memory" in data or "data" in data:
                return {"results": [data], "raw": data}
            return {"results": [], "raw": data}
        if isinstance(data, list):
            return {"results": data, "raw": data}
        return {"results": [], "raw": data}

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        session = await self._get_session()
        last_body = ""
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(f"{self.host}{path}", json=payload) as resp:
                        text = await resp.text()
                        last_body = text[:1000]
                        if resp.status >= 400:
                            raise RuntimeError(f"{resp.status} {resp.reason}: {last_body}")
                        if not text.strip():
                            return {}
                        return json.loads(text)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries - 1:
                    raise RuntimeError(f"POST {path} failed after {self.max_retries} attempts: {exc}") from exc
                sleep_for = self.retry_delay * (attempt + 1)
                print(f"Mem0 cloud POST {path} attempt {attempt + 1}/{self.max_retries} failed: {exc}", file=sys.stderr)
                await asyncio.sleep(sleep_for)
        raise RuntimeError(f"POST {path} failed: {last_body}")

    async def _get_json(self, path: str) -> Any:
        session = await self._get_session()
        last_body = ""
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.get(f"{self.host}{path}") as resp:
                        text = await resp.text()
                        last_body = text[:1000]
                        if resp.status >= 400:
                            raise RuntimeError(f"{resp.status} {resp.reason}: {last_body}")
                        if not text.strip():
                            return {}
                        return json.loads(text)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries - 1:
                    raise RuntimeError(f"GET {path} failed after {self.max_retries} attempts: {exc}") from exc
                print(f"Mem0 cloud GET {path} attempt {attempt + 1}/{self.max_retries} failed: {exc}", file=sys.stderr)
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"GET {path} failed: {last_body}")

    async def _wait_for_event(self, event_id: str) -> Dict[str, Any]:
        started = time.monotonic()
        while (time.monotonic() - started) < self.event_poll_timeout:
            data = await self._get_json(f"/v1/event/{event_id}/")
            status = str(data.get("status") or "").upper() if isinstance(data, dict) else ""
            if status == "SUCCEEDED":
                return data
            if status == "FAILED":
                raise RuntimeError(f"Mem0 cloud event {event_id} failed: {data}")
            await asyncio.sleep(self.event_poll_interval)
        raise TimeoutError(f"Mem0 cloud event {event_id} timed out after {self.event_poll_timeout:.0f}s")

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        observation_date: Optional[str] = None,
        timestamp: Optional[int] = None,
        custom_instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del observation_date
        payload: Dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "output_format": "v1.1",
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp
        if metadata:
            payload["metadata"] = metadata
        if custom_instructions:
            payload["custom_instructions"] = custom_instructions
        last_event_error: Optional[BaseException] = None
        for event_attempt in range(self.event_retries):
            data = await self._post_json("/v3/memories/add/", payload)
            if not isinstance(data, dict) or not data.get("event_id"):
                return self._normalise_add_response(data)

            event_id = str(data["event_id"])
            try:
                event_data = await self._wait_for_event(event_id)
                return self._normalise_add_response(event_data)
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_event_error = exc
                if event_attempt >= self.event_retries - 1:
                    break
                sleep_for = self.retry_delay * (event_attempt + 1)
                print(
                    f"Mem0 cloud add event {event_id} failed; retrying add "
                    f"{event_attempt + 2}/{self.event_retries}: {exc}",
                    file=sys.stderr,
                )
                await asyncio.sleep(sleep_for)
        raise RuntimeError(f"Mem0 cloud add event failed after {self.event_retries} attempts") from last_event_error

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query,
            "filters": {"user_id": user_id},
            "top_k": top_k,
            "rerank": rerank,
        }
        if score_debug:
            payload["score_debug"] = True
        data = await self._post_json("/v3/memories/search/", payload)
        if isinstance(data, dict):
            results = data.get("results") or data.get("memories") or []
        else:
            results = data
        if not isinstance(results, list):
            return []
        return sorted([item for item in results if isinstance(item, dict)], key=lambda item: item.get("score", 0), reverse=True)

    async def delete_user(self, user_id: str) -> bool:
        session = await self._get_session()
        try:
            async with self.limiter:
                async with session.delete(f"{self.host}/v1/entities/user/{user_id}/") as resp:
                    if resp.status in {404, 405}:
                        return False
                    resp.raise_for_status()
            return True
        except Exception:
            return False


def load_longmemeval(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON dataset, got {type(data)} from {path}")
    return data


def event_content(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages).strip()


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


def timestamp_to_iso(raw: object) -> Optional[str]:
    dt = parse_dt(raw) or parse_longmemeval_datetime(raw)
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_fragment(raw: object, *, max_len: int = 64) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "")).strip("._")
    return (clean or "sample")[:max_len]


def collect_official_write_events(entry: Dict[str, Any], official: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    qid = str(entry["question_id"])
    pair_order = 0
    sorted_sessions = official["sort_sessions_chronologically"](entry)
    for session_idx, (session_id, date_raw, session_turns) in enumerate(sorted_sessions):
        session_ts = normalize_timestamp(date_raw)
        session_unix = official["parse_longmemeval_date"](date_raw) if date_raw else None
        pairs = official["pair_turns"](session_turns or [])
        for pair_idx, messages in enumerate(pairs):
            clean_messages = [
                {"role": str(msg.get("role") or ""), "content": str(msg.get("content") or "")}
                for msg in messages
                if str(msg.get("role") or "") and str(msg.get("content") or "").strip()
            ]
            if not clean_messages:
                continue
            pair_order += 1
            content = event_content(clean_messages)
            write_id = make_write_id(
                agent=AGENT,
                question_id=qid,
                write_type=WRITE_TYPE,
                content=content,
                session_id=str(session_id),
                turn_span=[session_idx, pair_idx],
                timestamp=session_ts,
            )
            events.append(
                {
                    "write_id": write_id,
                    "session_id": str(session_id),
                    "session_index": session_idx,
                    "pair_index": pair_idx,
                    "turn_span": [session_idx, pair_idx],
                    "timestamp": session_ts,
                    "timestamp_iso": timestamp_to_iso(session_ts),
                    "timestamp_unix": session_unix,
                    "messages": clean_messages,
                    "content_text": content,
                    "original_index": pair_order - 1,
                }
            )
    return events


def sort_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(event: Dict[str, Any]):
        dt = parse_dt(event.get("timestamp"))
        if dt is None:
            return (1, str(event.get("timestamp") or ""), int(event.get("original_index", 0)))
        return (0, dt, int(event.get("original_index", 0)))

    return sorted(list(events), key=sort_key)


def apply_official_cf_spec(events: Iterable[Dict[str, Any]], spec) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    mutated: List[Dict[str, Any]] = []
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
        updated["timestamp_iso"] = timestamp_to_iso(updated["timestamp"])
        dt = parse_dt(updated["timestamp"])
        updated["timestamp_unix"] = int(dt.replace(tzinfo=timezone.utc).timestamp()) if dt else event.get("timestamp_unix")
        target_timestamp = updated["timestamp"]
        mutated.append(updated)
    return sort_events(mutated), target_timestamp


def build_write_records(qid: str, events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for write_order, event in enumerate(sort_events(events), start=1):
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
                origin="official_mem0_ingress",
            )
        )
    return records


def normalize_memory_text(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def extract_memory_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    if item.get("memory") is not None:
        return str(item.get("memory") or "")
    if item.get("data") is not None and not isinstance(item.get("data"), dict):
        return str(item.get("data") or "")
    data = item.get("data")
    if isinstance(data, dict):
        return str(data.get("memory") or data.get("text") or "")
    return str(item.get("text") or item.get("content") or "")


def metadata_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("metadata", "payload", "user_metadata", "memory_metadata"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    data = result.get("data")
    if isinstance(data, dict):
        for key in ("metadata", "payload"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
    return {}


def update_lineage_maps(
    *,
    response: Dict[str, Any],
    event: Dict[str, Any],
    native_to_source: Dict[str, List[str]],
    text_to_source: Dict[str, List[str]],
    add_results: List[Dict[str, Any]],
) -> None:
    results = response.get("results") if isinstance(response, dict) else []
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, dict):
            continue
        item_text = extract_memory_text(item)
        native_id = str(item.get("id") or item.get("memory_id") or "").strip()
        if native_id:
            existing = native_to_source.get(native_id, [])
            native_to_source[native_id] = normalize_list([*existing, event["write_id"]])
        norm_text = normalize_memory_text(item_text)
        if norm_text:
            existing = text_to_source.get(norm_text, [])
            text_to_source[norm_text] = normalize_list([*existing, event["write_id"]])
        add_results.append({**item, "source_write_id": event["write_id"]})


def result_source_ids(
    result: Dict[str, Any],
    native_to_source: Dict[str, List[str]],
    text_to_source: Dict[str, List[str]],
) -> List[str]:
    metadata = metadata_from_result(result)
    direct = normalize_list(metadata.get("source_write_ids") or [])
    if not direct and metadata.get("source_write_id"):
        direct = normalize_list([metadata.get("source_write_id")])
    if direct:
        return direct
    native_id = str(result.get("id") or result.get("memory_id") or "").strip()
    if native_id and native_to_source.get(native_id):
        return normalize_list(native_to_source[native_id])
    norm_text = normalize_memory_text(extract_memory_text(result))
    if norm_text and text_to_source.get(norm_text):
        return normalize_list(text_to_source[norm_text])
    return []


def normalize_search_result(
    result: Dict[str, Any],
    *,
    native_to_source: Dict[str, List[str]],
    text_to_source: Dict[str, List[str]],
    write_record_by_id: Dict[str, Dict[str, Any]],
    fill_created_at: bool,
) -> Dict[str, Any]:
    memory_text = extract_memory_text(result)
    entry = dict(result)
    entry["memory"] = memory_text
    if "score" not in entry:
        entry["score"] = result.get("similarity") or result.get("distance") or 0
    source_ids = result_source_ids(entry, native_to_source, text_to_source)
    entry["source_write_ids"] = source_ids
    if fill_created_at and not entry.get("created_at") and source_ids:
        source_ts = write_record_by_id.get(source_ids[0], {}).get("timestamp")
        source_iso = timestamp_to_iso(source_ts)
        if source_iso:
            entry["created_at"] = source_iso
    return entry


def build_item_records(
    *,
    search_results: Sequence[Dict[str, Any]],
    write_record_by_id: Dict[str, Dict[str, Any]],
    stage: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    write_ids: List[str] = []
    bridge_items: List[Dict[str, Any]] = []
    seen = set()
    for rank, result in enumerate(search_results, start=1):
        source_ids = normalize_list(result.get("source_write_ids"))
        memory_text = extract_memory_text(result)
        native_id = str(result.get("id") or result.get("memory_id") or "").strip()
        if not source_ids:
            bridge_items.append(
                {
                    "text": memory_text,
                    "source": "mem0_official_unmapped_memory",
                    "source_form": PRIMARY_SOURCE_FORM,
                    "native_memory_id": native_id,
                    "audit_eligible": False,
                }
            )
            continue
        dedupe_key = (stage, tuple(source_ids), native_id, memory_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_session_ids = normalize_list(
            write_record_by_id.get(write_id, {}).get("session_id") for write_id in source_ids
        )
        event_timestamps = [
            write_record_by_id.get(write_id, {}).get("timestamp")
            for write_id in source_ids
            if write_record_by_id.get(write_id, {}).get("timestamp")
        ]
        item = build_item_record(
            write_id=source_ids[0] if len(source_ids) == 1 else None,
            source_write_ids=source_ids,
            source_session_ids=source_session_ids,
            event_timestamps=event_timestamps,
            memory_timestamps=[result.get("created_at")] if result.get("created_at") else event_timestamps,
            stage=stage,
            rank=rank,
            score=result.get("score"),
            timestamp=result.get("created_at") or (event_timestamps[0] if event_timestamps else None),
            write_type=WRITE_TYPE,
            source_form=PRIMARY_SOURCE_FORM,
            audit_eligible=True,
            text=memory_text,
            source=stage,
            extra={
                "native_memory_id": native_id,
                "metadata": metadata_from_result(result),
                "score_debug": result.get("score_debug"),
            },
        )
        items.append(item)
        write_ids.extend(item.get("source_write_ids") or [])
    return items, normalize_list(write_ids), bridge_items


def strip_mem_thinking(text: str) -> str:
    cleaned = re.sub(
        r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]",
        "",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if "ANSWER:" in cleaned:
        cleaned = cleaned.rsplit("ANSWER:", 1)[-1].strip()
    return cleaned


async def generate_official_answer(
    *,
    entry: Dict[str, Any],
    search_results: Sequence[Dict[str, Any]],
    answerer: Any,
    official: Dict[str, Any],
    answer_cutoff: int,
) -> str:
    question = str(entry.get("question") or "")
    question_date = str(entry.get("question_date") or "")
    question_date_human = official["parse_longmemeval_date_human"](question_date) if question_date else ""
    sliced = list(search_results[:answer_cutoff])
    sliced_chrono = sorted(sliced, key=lambda x: x.get("created_at") or "")
    prompt = official["get_answer_generation_prompt"](
        question=question,
        search_results=sliced_chrono,
        question_date=question_date_human,
        user_profile=None,
    )
    answer = await answerer.generate(system="", user=prompt)
    return strip_mem_thinking(answer)


async def ingest_events(
    *,
    mem0: Any,
    events: Sequence[Dict[str, Any]],
    user_id: str,
    qid: str,
    sample_tag: str,
    fail_fast: bool,
    show_progress: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[Dict[str, Any]], int, int]:
    native_to_source: Dict[str, List[str]] = {}
    text_to_source: Dict[str, List[str]] = {}
    add_results: List[Dict[str, Any]] = []
    processed = 0
    failed = 0
    ordered_events = sort_events(events)
    iterator: Iterable[Dict[str, Any]]
    if show_progress:
        iterator = tqdm(
            ordered_events,
            total=len(ordered_events),
            desc=f"Ingest {qid} {sample_tag}",
            unit="write",
            leave=False,
        )
    else:
        iterator = ordered_events
    for event in iterator:
        metadata = {
            "source_write_id": event["write_id"],
            "source_write_ids": [event["write_id"]],
            "question_id": qid,
            "source_session_id": event.get("session_id"),
            "source_turn_span": event.get("turn_span") or [],
            "source_event_order": event.get("original_index"),
            "source_timestamp": event.get("timestamp"),
        }
        response = await mem0.add(
            event["messages"],
            user_id=user_id,
            timestamp=event.get("timestamp_unix"),
            metadata=metadata,
        )
        if response is None:
            failed += 1
            if fail_fast:
                raise RuntimeError(f"Mem0 add failed for qid={qid} write={event['write_id']}")
        else:
            update_lineage_maps(
                response=response,
                event=event,
                native_to_source=native_to_source,
                text_to_source=text_to_source,
                add_results=add_results,
            )
        processed += 1
    return native_to_source, text_to_source, add_results, processed, failed


async def run_replay(
    *,
    entry: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    mem0: Any,
    answerer: Any,
    official: Dict[str, Any],
    args: argparse.Namespace,
    sample_tag: str,
) -> Dict[str, Any]:
    qid = str(entry["question_id"])
    qtype = str(entry.get("question_type") or "unknown")
    user_id = f"lme_{sanitize_fragment(qid)}_{sanitize_fragment(sample_tag, max_len=48)}"
    write_records = build_write_records(qid, events)
    write_record_by_id = {record["write_id"]: record for record in write_records}

    native_to_source, text_to_source, add_results, processed, failed = await ingest_events(
        mem0=mem0,
        events=events,
        user_id=user_id,
        qid=qid,
        sample_tag=sample_tag,
        fail_fast=args.fail_fast,
        show_progress=args.show_ingest_progress,
    )

    search_start = time.monotonic()
    raw_search_results = await mem0.search(
        str(entry.get("question") or ""),
        user_id=user_id,
        top_k=args.top_k,
        rerank=args.rerank,
        score_debug=args.score_debug,
    )
    search_latency_ms = (time.monotonic() - search_start) * 1000.0
    if not isinstance(raw_search_results, list):
        raw_search_results = []
    search_results = [
        normalize_search_result(
            result,
            native_to_source=native_to_source,
            text_to_source=text_to_source,
            write_record_by_id=write_record_by_id,
            fill_created_at=args.fill_created_at_from_source,
        )
        for result in raw_search_results
        if isinstance(result, dict)
    ]
    search_results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

    answer = await generate_official_answer(
        entry=entry,
        search_results=search_results,
        answerer=answerer,
        official=official,
        answer_cutoff=args.answer_cutoff,
    )

    retrieved_items, retrieved_write_ids, bridge_items = build_item_records(
        search_results=search_results,
        write_record_by_id=write_record_by_id,
        stage=PRIMARY_STAGE,
    )
    prompt_search_results = sorted(search_results[: args.answer_cutoff], key=lambda item: item.get("created_at") or "")
    prompt_items, prompt_write_ids, prompt_bridge_items = build_item_records(
        search_results=prompt_search_results,
        write_record_by_id=write_record_by_id,
        stage="mem0_official_answer_prompt",
    )
    bridge_items.extend(prompt_bridge_items)

    candidate_write_ids = [record["write_id"] for record in write_records]
    query_record = build_query_record(
        agent=AGENT,
        question_id=qid,
        question_type=qtype,
        query_time=entry.get("question_date"),
        question_date_used=entry.get("question_date"),
        baseline_answer=answer,
        candidate_write_ids=candidate_write_ids,
        retrieved_write_ids=retrieved_write_ids,
        selected_write_ids=prompt_write_ids,
        prompt_write_ids=prompt_write_ids,
        retrieved_items=retrieved_items,
        prompt_items=prompt_items,
        bridge_items=bridge_items,
        extra={
            "official_mem0_backend": args.backend,
            "official_project_name": args.project_name,
            "official_answerer_model": args.answerer_model,
            "official_top_k": args.top_k,
            "official_answer_cutoff": args.answer_cutoff,
            "official_rerank": args.rerank,
            "official_score_debug": args.score_debug,
            "official_search_latency_ms": round(search_latency_ms, 1),
            "official_user_id": user_id,
        },
    )
    trace_obj = {
        "question_id": qid,
        "question_type": qtype,
        "official_mem0_backend": args.backend,
        "official_user_id": user_id,
        "n_ingested_pairs": processed,
        "n_failed_pairs": failed,
        "n_add_results": len(add_results),
        "n_search_results": len(search_results),
        "search_latency_ms": round(search_latency_ms, 1),
        "add_results": add_results,
        "search_results": search_results,
    }
    return {
        "hypothesis": answer,
        "trace": trace_obj,
        "query_record": query_record,
        "write_records": write_records,
        "events": sort_events(events),
        "user_id": user_id,
    }


def filter_specs(specs: Sequence[Any], rule_mode: str) -> List[Any]:
    if rule_mode == "rollback-only":
        return [spec for spec in specs if getattr(spec, "rule_id", "") == "rollback_skip"]
    return list(specs)


async def maybe_cleanup(mem0: Any, user_id: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        await mem0.delete_user(user_id)
    except Exception:
        pass


async def async_main() -> None:
    args = parse_args()
    load_env_file([REPO_ROOT / ".env", Path.cwd() / ".env"], override=False)

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    desired_openai_api_key = os.getenv("OPENAI_API_KEY")
    desired_openai_base_url = os.getenv("OPENAI_BASE_URL")
    desired_mem0_api_key = os.getenv("MEM0_API_KEY")
    if args.backend == "cloud" and not os.getenv("MEM0_API_KEY"):
        raise RuntimeError("Official Mem0 cloud mode requires MEM0_API_KEY or --mem0-api-key.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY or --openai-api-key for official answer generation.")

    benchmarks_dir = ensure_memory_benchmarks_repo(
        args.memory_benchmarks_dir,
        auto_clone=args.auto_clone_memory_benchmarks,
        ref=args.memory_benchmarks_ref,
    )
    official = import_official_modules(benchmarks_dir)
    # The official runner imports dotenv with override=True. Restore the shard-selected
    # keys immediately after import so parallel runs keep using OPENAI_API_KEY_N.
    if desired_openai_api_key:
        os.environ["OPENAI_API_KEY"] = desired_openai_api_key
    if desired_openai_base_url:
        os.environ["OPENAI_BASE_URL"] = desired_openai_base_url
    if desired_mem0_api_key:
        os.environ["MEM0_API_KEY"] = desired_mem0_api_key
    questions = load_longmemeval(args.longmemeval_file)
    if args.offset:
        questions = questions[args.offset :]
    if args.limit:
        questions = questions[: args.limit]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("", encoding="utf-8")
    if args.trace_jsonl:
        args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.trace_jsonl.write_text("", encoding="utf-8")
    audit_query_path, audit_write_path = derive_audit_paths(args.trace_jsonl)
    cf_run_path, cf_query_path = derive_cf_paths(args.trace_jsonl)
    for path in (audit_query_path, audit_write_path, cf_run_path, cf_query_path):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    run_id = args.run_id or datetime.now().strftime("%m%d%H%M%S")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "agent": AGENT,
                    "backend": args.backend,
                    "questions": len(questions),
                    "memory_benchmarks_dir": str(benchmarks_dir),
                    "memory_benchmarks_ref": args.memory_benchmarks_ref,
                    "top_k": args.top_k,
                    "answer_cutoff": args.answer_cutoff,
                    "enable_cf_wrapper": args.enable_cf_wrapper,
                    "cf_rule_mode": args.cf_rule_mode,
                    "cf_max_writes": args.cf_max_writes,
                },
                indent=2,
            )
        )
        return

    LLMClient = official["LLMClient"]
    if args.backend == "cloud":
        mem0 = CurrentMem0CloudClient(
            host=args.mem0_host,
            api_key=os.getenv("MEM0_API_KEY") or "",
            max_retries=args.max_add_retries,
            event_retries=args.add_event_retries,
            retry_delay=args.retry_delay,
            timeout=args.request_timeout,
            rpm=args.rpm,
            event_poll_timeout=args.event_poll_timeout,
        )
    else:
        Mem0Client = official["Mem0Client"]
        mem0 = Mem0Client(
            mode=args.backend,
            host=args.mem0_host,
            api_key=None,
            max_retries=args.max_add_retries,
            retry_delay=args.retry_delay,
            rpm=args.rpm,
            timeout=args.request_timeout,
            event_poll_timeout=args.event_poll_timeout,
        )
    answerer = LLMClient(
        model=args.answerer_model,
        provider=args.provider,
        rpm=args.rpm,
        base_url=args.openai_base_url,
    )

    async with mem0:
        pbar = tqdm(questions, total=len(questions), desc="MEM0-OFFICIAL", unit="q")
        for idx, entry in enumerate(pbar, start=1):
            qid = str(entry["question_id"])
            events = collect_official_write_events(entry, official)
            sample_tag = f"base_{run_id}_{idx:04d}"
            baseline = await run_replay(
                entry=entry,
                events=events,
                mem0=mem0,
                answerer=answerer,
                official=official,
                args=args,
                sample_tag=sample_tag,
            )
            with args.out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"question_id": qid, "hypothesis": baseline["hypothesis"]}, ensure_ascii=False) + "\n")
            append_audit_jsonl(args.trace_jsonl, baseline["trace"])
            for write_record in baseline["write_records"]:
                append_audit_jsonl(audit_write_path, write_record)
            append_audit_jsonl(audit_query_path, baseline["query_record"])

            if args.enable_cf_wrapper:
                specs = filter_specs(
                    build_cf_specs(
                        question_type=str(entry.get("question_type") or "unknown"),
                        query_record=baseline["query_record"],
                        write_records=baseline["write_records"],
                        answer_session_ids=entry.get("answer_session_ids", []),
                        max_writes=args.cf_max_writes,
                        scope=args.cf_target_scope,
                    ),
                    args.cf_rule_mode,
                )
                cf_results = []
                for spec_idx, spec in enumerate(specs, start=1):
                    mutated_events, target_timestamp = apply_official_cf_spec(events, spec)
                    outcome = await run_replay(
                        entry=entry,
                        events=mutated_events,
                        mem0=mem0,
                        answerer=answerer,
                        official=official,
                        args=args,
                        sample_tag=f"cf_{run_id}_{idx:04d}_{spec_idx:02d}_{spec.rule_id}_{spec.target_write_id[-8:]}",
                    )
                    cf_results.append(
                        {
                            "spec": spec,
                            "cf_answer": outcome["hypothesis"],
                            "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                            "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                            "target_timestamp": target_timestamp,
                            "cf_extra": {
                                "n_search_results": outcome["trace"].get("n_search_results"),
                                "n_add_results": outcome["trace"].get("n_add_results"),
                                "official_user_id": outcome.get("user_id"),
                            },
                        }
                    )
                    await maybe_cleanup(mem0, outcome["user_id"], args.cleanup_users)

                run_records, query_summary = summarize_replay_cf(
                    agent=AGENT,
                    entry=entry,
                    baseline_query_record=baseline["query_record"],
                    write_records=baseline["write_records"],
                    cf_results=cf_results,
                    dominance_threshold=args.cf_dominance_threshold,
                )
                append_cf_outputs(run_path=cf_run_path, query_path=cf_query_path, run_records=run_records, query_summary=query_summary)

            await maybe_cleanup(mem0, baseline["user_id"], args.cleanup_users)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
