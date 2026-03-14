#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longmemeval_unified_answer import EvidenceRow, build_unified_qa_messages


@dataclass
class ShareMemory:
    p1: List[str] = field(default_factory=list)
    p2: List[str] = field(default_factory=list)
    t1: List[str] = field(default_factory=list)
    t2: List[str] = field(default_factory=list)
    shared: List[str] = field(default_factory=list)
    mutual: List[str] = field(default_factory=list)


@dataclass
class EpisodeSessionInformation:
    p1: List[str] = field(default_factory=list)
    p2: List[str] = field(default_factory=list)
    t1: List[str] = field(default_factory=list)
    t2: List[str] = field(default_factory=list)
    shared: List[str] = field(default_factory=list)
    mutual: List[str] = field(default_factory=list)


@dataclass
class EpisodeSessionData:
    s1_name: str
    s2_name: str
    information: EpisodeSessionInformation


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


def normalize_list(items: object, max_items: int) -> List[str]:
    if not isinstance(items, list):
        return []
    normalized: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        line = re.sub(r"\s+", " ", item).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(line)
        if max_items > 0 and len(normalized) >= max_items:
            break
    return normalized


def normalize_memory(memory_raw: Dict, max_items: int) -> ShareMemory:
    return ShareMemory(
        p1=normalize_list(memory_raw.get("p1", []), max_items),
        p2=normalize_list(memory_raw.get("p2", []), max_items),
        t1=normalize_list(memory_raw.get("t1", []), max_items),
        t2=normalize_list(memory_raw.get("t2", []), max_items),
        shared=normalize_list(memory_raw.get("shared", []), max_items),
        mutual=normalize_list(memory_raw.get("mutual", []), max_items),
    )


def memory_from_session_info(info: object, max_items: int) -> ShareMemory:
    return canonicalize_memory(
        ShareMemory(
            p1=list(getattr(info, "p1", []) or []),
            p2=list(getattr(info, "p2", []) or []),
            t1=list(getattr(info, "t1", []) or []),
            t2=list(getattr(info, "t2", []) or []),
            shared=list(getattr(info, "shared", []) or []),
            mutual=list(getattr(info, "mutual", []) or []),
        ),
        max_items=max_items,
        retain_mutual=False,
    )


def canonicalize_memory(memory: ShareMemory, max_items: int, retain_mutual: bool) -> ShareMemory:
    # EPISODE treats mutual events as session-local and migrates them into shared memory.
    shared_seed = list(memory.shared) + list(memory.mutual)
    return ShareMemory(
        p1=normalize_list(memory.p1, max_items),
        p2=normalize_list(memory.p2, max_items),
        t1=normalize_list(memory.t1, max_items),
        t2=normalize_list(memory.t2, max_items),
        shared=normalize_list(shared_seed, max_items),
        mutual=normalize_list(memory.mutual, max_items) if retain_mutual else [],
    )


def format_pairs_as_dialogue(pairs: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for user_turn, assistant_turn in pairs:
        user_turn = user_turn.strip()
        assistant_turn = assistant_turn.strip()
        if user_turn:
            lines.append(f"USER: {user_turn}")
        if assistant_turn:
            lines.append(f"ASSISTANT: {assistant_turn}")
    return "\n".join(lines)


def flatten_memory_candidates(memory: ShareMemory, include_mutual: bool) -> List[str]:
    candidates: List[str] = []
    tagged_items: List[Tuple[str, List[str]]] = [
        ("P1", memory.p1),
        ("P2", memory.p2),
        ("T1", memory.t1),
        ("T2", memory.t2),
        ("SHARED", memory.shared),
    ]
    if include_mutual:
        tagged_items.append(("MUTUAL", memory.mutual))
    for tag, items in tagged_items:
        for item in items:
            candidates.append(f"[{tag}] {item}")
    return candidates


def flatten_memory_candidates_raw(memory: ShareMemory, include_mutual: bool) -> List[str]:
    candidates: List[str] = []
    raw_lists: List[List[str]] = [memory.p1, memory.t1, memory.p2, memory.t2, memory.shared]
    if include_mutual:
        raw_lists.append(memory.mutual)
    for bucket in raw_lists:
        for item in bucket:
            clean = item.strip()
            if clean:
                candidates.append(clean)
    return normalize_list(candidates, max_items=0)


def memory_item_count(memory: ShareMemory) -> int:
    return (
        len(memory.p1)
        + len(memory.p2)
        + len(memory.t1)
        + len(memory.t2)
        + len(memory.shared)
        + len(memory.mutual)
    )


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def lexical_topk(question: str, candidates: List[str], k: int) -> List[str]:
    if not candidates or k <= 0:
        return []
    q_tokens = set(simple_tokenize(question))
    scored: List[Tuple[float, str]] = []
    for cand in candidates:
        c_tokens = set(simple_tokenize(cand))
        if not c_tokens:
            score = 0.0
        else:
            overlap = len(q_tokens & c_tokens)
            score = overlap / max(1, len(q_tokens))
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [cand for score, cand in scored[:k] if score > 0]
    if selected:
        return selected
    return candidates[:k]


class OpenAIJsonClient:
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

    def _chat(self, messages: List[Dict], response_format: Optional[Dict] = None) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return text.strip()

    @staticmethod
    def _parse_json_lenient(text: str) -> Dict:
        candidates: List[str] = [text.strip()]
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
            candidates.append(stripped.strip())

        start = stripped.find("{")
        end = stripped.rfind("}")
        if 0 <= start < end:
            candidates.append(stripped[start : end + 1].strip())

        for candidate in candidates:
            if not candidate:
                continue
            normalized = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                payload = json.loads(normalized)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        raise ValueError("No valid JSON object parsed from model output.")

    def chat_json(self, messages: List[Dict], retries: int = 1) -> Dict:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                text = self._chat(messages, response_format={"type": "json_object"})
                return self._parse_json_lenient(text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.4 + 0.4 * attempt)
        raise RuntimeError(f"JSON generation failed: {last_error}")

    def chat_text(self, messages: List[Dict], retries: int = 2) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return self._chat(messages, response_format=None)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"Text generation failed: {last_error}")


def extract_session_memory(
    llm: OpenAIJsonClient,
    dialogue_text: str,
    max_items: int,
    dry_run: bool,
    json_retries: int,
) -> ShareMemory:
    if dry_run:
        return ShareMemory()

    system_prompt = (
        "You are a conversation memory extractor. "
        "Extract only stable facts and meaningful events from this session."
    )
    user_prompt = (
        "Given this dialogue between USER and ASSISTANT, extract memory in JSON with keys:\n"
        "p1, p2, t1, t2, shared, mutual.\n"
        "- p1: USER persona facts\n"
        "- p2: ASSISTANT persona facts\n"
        "- t1: USER temporary events\n"
        "- t2: ASSISTANT temporary events\n"
        "- shared: shared memories mentioned as past joint experiences\n"
        "- mutual: notable events in this current session between USER and ASSISTANT\n"
        "Rules: each key must be an array of concise sentences, no duplicates, max 1 sentence per item.\n\n"
        f"Dialogue:\n{dialogue_text}\n"
    )
    try:
        payload = llm.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            retries=json_retries,
        )
        return normalize_memory(payload, max_items)
    except Exception:
        return ShareMemory()


def merge_share_memory_fallback(
    previous: ShareMemory,
    current: ShareMemory,
    max_items: int,
    retain_mutual: bool,
) -> ShareMemory:
    def merge(new_items: List[str], old_items: List[str]) -> List[str]:
        return normalize_list(new_items + old_items, max_items)

    merged = ShareMemory(
        p1=merge(current.p1, previous.p1),
        p2=merge(current.p2, previous.p2),
        t1=merge(current.t1, previous.t1),
        t2=merge(current.t2, previous.t2),
        shared=merge(current.shared + current.mutual, previous.shared),
        mutual=normalize_list(current.mutual, max_items) if retain_mutual else [],
    )
    return canonicalize_memory(merged, max_items=max_items, retain_mutual=retain_mutual)


def update_share_memory(
    llm: OpenAIJsonClient,
    previous: ShareMemory,
    current: ShareMemory,
    max_items: int,
    dry_run: bool,
    json_retries: int,
    retain_mutual: bool,
) -> ShareMemory:
    if dry_run:
        return previous

    system_prompt = (
        "You are a memory manager for a dialogue agent. "
        "Merge and clean memory while preserving useful long-term facts."
    )
    user_prompt = (
        "Merge PREVIOUS and CURRENT memory into UPDATED memory in JSON with keys:\n"
        "p1, p2, t1, t2, shared, mutual (all arrays).\n"
        "Guidelines:\n"
        "1) Remove duplicates and low-information fragments.\n"
        "2) For conflicting events, keep the newer valid state and express transition when needed.\n"
        "3) Keep temporary events concise and remove trivial short-lived noise.\n"
        "4) Convert mutual events into shared memory when appropriate.\n"
        "5) Keep memory coherent for future QA.\n\n"
        f"PREVIOUS={json.dumps(asdict(previous), ensure_ascii=False)}\n"
        f"CURRENT={json.dumps(asdict(current), ensure_ascii=False)}\n"
    )
    try:
        payload = llm.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            retries=json_retries,
        )
        merged = normalize_memory(payload, max_items)
        return canonicalize_memory(merged, max_items=max_items, retain_mutual=retain_mutual)
    except Exception:
        return merge_share_memory_fallback(
            previous,
            current,
            max_items=max_items,
            retain_mutual=retain_mutual,
        )


def select_memories(
    llm: OpenAIJsonClient,
    question: str,
    recent_dialogue: str,
    candidates: List[str],
    top_k: int,
    dry_run: bool,
    json_retries: int,
) -> List[str]:
    if not candidates:
        return []
    if dry_run:
        return lexical_topk(question, candidates, top_k)

    indexed = "\n".join(f"{idx}. {item}" for idx, item in enumerate(candidates))
    system_prompt = "You select the most relevant memory items for answering a user question."
    user_prompt = (
        "Return JSON: {\"selected_indices\": [int, ...]} only.\n"
        f"Select up to {top_k} indices.\n"
        "Prefer factual memory that directly helps answer.\n"
        "If nothing relevant, return empty list.\n\n"
        f"Question:\n{question}\n\n"
        f"Recent Dialogue:\n{recent_dialogue}\n\n"
        f"Candidates:\n{indexed}\n"
    )

    try:
        payload = llm.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            retries=json_retries,
        )
        raw = payload.get("selected_indices", [])
        indices: List[int] = []
        for item in raw:
            if isinstance(item, int):
                indices.append(item)
        unique_indices = []
        seen = set()
        for idx in indices:
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
            if len(unique_indices) >= top_k:
                break
        if unique_indices:
            return [candidates[idx] for idx in unique_indices]
    except Exception:
        pass

    return lexical_topk(question, candidates, top_k)


def answer_question(
    llm: OpenAIJsonClient,
    question: str,
    query_with_date: str,
    selected_memories: List[str],
    recent_dialogue: str,
    dry_run: bool,
    force_abstain_when_uncertain: bool,
) -> str:
    if dry_run:
        return "DRY_RUN_PLACEHOLDER"

    del question, recent_dialogue, force_abstain_when_uncertain
    evidence_rows = [EvidenceRow(text=item, source="selected") for item in selected_memories]
    return llm.chat_text(build_unified_qa_messages(query_with_date, evidence_rows)).strip()


def call_prompt_text(llm: OpenAIJsonClient, prompt: str, retries: int = 2) -> str:
    return llm.chat_text([{"role": "user", "content": prompt}], retries=retries)


def episode_extraction_prompt(speaker1: str, speaker2: str, dialogues_text: str) -> str:
    return f"""You are a conversation analyst tasked with examining two conversations.

In your analysis, categorize the dialogue based on five criteria:
1. **Persona Information**: Discuss aspects such as personality, job, age, education, favorite foods, music, hobbies, family life, daily activities, health, etc.
2. **Temporary event**: Identify information that will soon become irrelevant, such as upcoming deadlines like "I need to submit my assignment by Friday" or temporary states like "I have a cold."
3. **Shared Memory**: Focus on past experiences that the speakers refer to during their conversation, which they have previously experienced together. This category includes both explicitly mentioned memories and those implied through their dialogue.
4. **Mutual Event**: This category captures significant events and interactions occurring directly between {speaker1} and {speaker2} during the current conversation, excluding any third-party involvement.
5. **None**: Assign this category to parts of the conversation that do not fit into the above categories.

Proceed to analyze the dialogue, addressing it one turn at a time:
{dialogues_text}

Your task is to extract:
- Persona information for {speaker1}
- Persona information for {speaker2}
- Temporary event for {speaker1}
- Temporary event for {speaker2}
- Shared memories between {speaker1} and {speaker2}
- Mutual events occurring during the conversation between {speaker1} and {speaker2}

Format your findings by separating each category with '***'. If no information is found for a category, indicate it with 'None'. The expected format is:
[***Persona: {speaker1}'s information or 'None'***Persona: {speaker2}'s information or 'None'***Temporary: {speaker1}'s event or 'None'***Temporary: {speaker2}'s event or 'None'***Shared Memory: information or 'None'***Mutual Event: information or 'None'***]
Present your responses directly, using the speakers' names without pronouns and avoiding category labels. For instance, rather than stating "***{speaker1}'s temporary event includes an upcoming math project due tomorrow.***", simply note "***Temporary: {speaker1} has a math project due tomorrow.***"
Ensure that each analysis output is succinct, covering only the essential elements of the dialogue. Ensure you cover every part of the dialogue comprehensively.
If a specific category does not apply, move on to the next without mention. Your detailed analysis will help illuminate the nuances of their interactions, capturing the essence of their shared and immediate experiences within the current dialogue.
Answer:"""


def episode_return_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Jr|Sr|Prof|MR|MRS|MS|DR|JR|SR|PROF|COL|SGT|LT|CPL)\.",
        r"\1<dot>",
        text,
    )
    text = re.sub(r"\b([A-Za-z])\.([A-Za-z])\.", r"\1<dot>\2<dot>", text)
    sentences = re.compile(r"(?<=\.|\?|!)\s").split(text.strip())
    sentences = [sentence.replace("<dot>", ".") for sentence in sentences]
    exclusion_phrases = [
        "There is no",
        "There are no",
        "information is not",
        "information cannot be",
        "None",
        "No shared",
        "No temporal information",
        "no temporal information",
        "no information for",
        "No specific information about",
    ]
    return [
        sentence
        for sentence in sentences
        if sentence and not any(phrase in sentence for phrase in exclusion_phrases)
    ]


def episode_split_information(text: str) -> Dict[str, List[str]]:
    patterns = {
        "p1": r"\*\*\*Persona: (.*?)\*\*\*",
        "p2": r"\*\*\*Persona:.*?\*\*\*Persona: (.*?)\*\*\*",
        "t1": r"\*\*\*Temporary: (.*?)\*\*\*",
        "t2": r"\*\*\*Temporary:.*?\*\*\*Temporary: (.*?)\*\*\*",
        "shared": r"\*\*\*Shared Memory: (.*?)\*\*\*",
        "mutual": r"\*\*\*Mutual Event: (.*?)(?:\*\*\*|\*\*\*\]|\*+)$",
    }
    extracted: Dict[str, List[str]] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        extracted[key] = episode_return_sentences(match.group(1)) if match else []
    return extracted


def episode_process_extraction(session_data: EpisodeSessionData, extracted_information: str) -> EpisodeSessionData:
    parsed = episode_split_information(extracted_information)
    session_data.information = EpisodeSessionInformation(
        p1=parsed["p1"],
        p2=parsed["p2"],
        t1=parsed["t1"],
        t2=parsed["t2"],
        shared=parsed["shared"],
        mutual=parsed["mutual"],
    )
    return session_data


def episode_return_memory(info: EpisodeSessionData) -> str:
    def process_list(lst: List[str]) -> str:
        return "\n".join(["- " + x for x in lst]) if lst else ""

    return f"""Persona:
{process_list(info.information.p1)}
{process_list(info.information.p2)}
Personal event:
{process_list(info.information.t1)}
{process_list(info.information.t2)}
Shared memory:
{process_list(info.information.shared)}
Mutual event:
{process_list(info.information.mutual)}"""


def episode_update_prompt(previous_memory: str, current_memory: str) -> str:
    return f"""\
You are a language expert who understands the flow of conversation and manages memory. To effectively manage memory in a conversational system, it is crucial to understand the memory itself. As the conversation progresses, compare the information from previous sessions with the current session to update the memory and remove unnecessary sentences. Memory is categorized into the following four types:
1. Persona information: This captures essential characteristics, including personality, occupation, and interests.
2. Personal event: This information covers transient details like impending deadlines or current health conditions.
3. Mutual event: This captures significant interactions between the speakers, focusing on substantial events directly involving both individuals. Over time, these mutual events become new shared memories.
4. Shared memory: This refers to past experiences or memories that the two speakers have shared together prior to the current conversational context.
Guidelines for Memory Management:
Tasks to Perform in the Current Session:
1. Remove incomplete information: Remove sentences that are incomplete or do not clearly convey the context.
    * Example: "SAM is interested in something." or "SAM mentions a place he visited."
2. Remove information not suitable for conversation topics: Remove information that is irrelevant to the main topic of conversation.
    * Example: "JANE remembers SAM." or “JANE has a need to urinate.”
3. Remove unrelated personal events: Remove personal event information that is not directly related to the individual or does not influence the conversation flow.
    * Example: "MARK talked about a coworker who went on vacation last month."
4. Remove duplicate information: If the same information is provided in both Persona and Personal events, or if the same information is provided in Persona and Shared memory, remove the Persona and retain the other information.
    * Example: “KATE enjoys watching movies.” (Persona) and “KATE often watches movies on weekends.” (Personal event) provide similar information, so remove the Persona.
    * Example: “MIKE remembers the trip to Paris.” (Persona) and “MIKE and JANE shared a memorable trip to Paris.” (Shared memory) are similar; remove the Persona.
5. Update Persona based on Mutual events: Update the Persona with emotions or reactions caused by Mutual events, and write sentences in the past tense.
    * Example: The Persona "JACK feels betrayed and angry." should be updated to "SARAH told JACK about her secret involvement in a rival project, causing JACK to feel betrayed and angry."
Methods for Memory Update:
1. Connect sequential/causal events: Link and update events that are sequential or have a causal relationship.
    * Example:
        * Previous memory:
            * Tom recently got a new job.
            * Tom was very nervous on his first day at work.
        * Current memory:
            * Tom successfully completed his first project at the new job.
        * Updated memory:
            * Tom recently got a new job and was very nervous on his first day.
            * Tom has since successfully completed his first project.
2. Update conflicting events: Reflect changes or transitions when the previous and current memories contain conflicting information.
    * Example:
        * Previous memory:
            * Ellie did not enjoy her recent trip.
            * Ellie said she would no longer plan trips.
        * Current memory:
            * Ellie is planning a trip with her friends.
            * Ellie is looking forward to traveling again.
        * Updated memory:
            * Ellie did not enjoy her recent trip, but now she is planning a new trip with friends and is looking forward to it.
3. Remove unnecessary personal event information: Exclude any unnecessary details about personal events. If the personal event only reflects a very short-term, trivial state (such as someone being in transit), it should be removed.
• Example: "Jay is on the bus" should be removed.
4. Accumulate unrelated events: Accumulate personal events that do not fit guidelines 1 through 3.
    * Example:
        * Previous memory:
            * JANE likes spicy food.
        * Current memory:
            * JANE dislikes math.
        * Updated memory:
            * JANE likes spicy food.
            * JANE dislikes math.
5. Use the past tense for Mutual events: Mutual events from the current session become past events, so convert them to the past tense.
    * Example:
        * Previous memory:
            * John and Alice are planning a trip together.
        * Current memory:
            * John and Alice have finalized the details of their trip.
        * Updated memory:
            * John and Alice planned a trip together and have finalized the details.
Actual Content Update:
Use the following structure to update the memory based on the provided guidelines.
All sentences in the updated memory must start with a person’s name.
Previous memory:
{previous_memory}
Current memory:
{current_memory}
Updated memory:"""


def episode_process_update_prompt(previous_data: EpisodeSessionData, current_data: EpisodeSessionData) -> str:
    return episode_update_prompt(episode_return_memory(previous_data), episode_return_memory(current_data))


def episode_replace_names_with_uppercase(response: str, name1: str, name2: str) -> str:
    pattern = re.compile(r"\b(" + re.escape(name1.lower()) + r"|" + re.escape(name2.lower()) + r")\b", re.IGNORECASE)

    def replace_name(match: re.Match) -> str:
        matched_name = match.group(0)
        if matched_name.lower() == name1.lower():
            return name1
        if matched_name.lower() == name2.lower():
            return name2
        return matched_name

    return pattern.sub(replace_name, response)


def episode_parse_memory(update_response: str, s1_name: str, s2_name: str) -> EpisodeSessionData:
    persona_info: Dict[str, List[str]] = {}
    personal_event_info: Dict[str, List[str]] = {}
    shared_memory_list: List[str] = []
    current_category: Optional[str] = None

    if update_response.startswith("Updated memory :"):
        update_response = update_response[len("Updated memory :") :].strip()

    for line in update_response.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_category = line[:-1].strip().lower()
            continue
        if re.match(r"^-\s?", line):
            sentence = line[1:].strip()
            if current_category == "persona":
                name = sentence.split()[0].upper()
                persona_info.setdefault(name, []).append(sentence)
            elif current_category == "personal event":
                name = sentence.split()[0].upper()
                personal_event_info.setdefault(name, []).append(sentence)
            elif current_category in {"shared memory", "mutual event"}:
                shared_memory_list.append(sentence)

    return EpisodeSessionData(
        s1_name=s1_name,
        s2_name=s2_name,
        information=EpisodeSessionInformation(
            p1=persona_info.get(s1_name, []),
            p2=persona_info.get(s2_name, []),
            t1=personal_event_info.get(s1_name, []),
            t2=personal_event_info.get(s2_name, []),
            shared=shared_memory_list,
            mutual=[],
        ),
    )


def episode_selection_prompt(
    data: EpisodeSessionData,
    dia_no_tag_text: str,
    next_speaker: str,
    include_mutual: bool,
) -> str:
    info_list = [data.information.p1, data.information.t1, data.information.p2, data.information.t2, data.information.shared]
    if include_mutual:
        info_list.append(data.information.mutual)
    candidates = ["Everyday Language"] + [item for info in info_list if info is not None for item in info]
    random.shuffle(candidates)
    candidates_string = "\n".join(f"- {value}" for value in candidates)
    return f"""\
You are a conversation analyst. \
You need to understand the context well and predict the next part of the dialogue.
Based on the provided candidate memories and dialogue history, select all the appropriate memories for the next part of the conversation. \
These memories are elements that form the basis of the conversation.
If no suitable memories are available, choose 'Everyday Language,' which refers to common, everyday expressions.

Task:
Candidate Memories:
{candidates_string}

Dialogue History:
{dia_no_tag_text} 
Select all the appropriate memories for the next part of the conversation by {next_speaker}. \
If there are two or more memories, separate them with '###':"""


def episode_qa_selection_prompt(
    question: str,
    dia_no_tag_text: str,
    candidates: List[str],
    top_k: int,
) -> str:
    candidates_string = "\n".join(f"- {value}" for value in candidates)
    return f"""\
You are a memory retrieval analyst for question answering.
Given candidate memories and recent dialogue context, select the memories that are most useful for answering the question.
Only select evidence that directly supports the answer.

Task:
Question:
{question}

Recent Dialogue:
{dia_no_tag_text}

Candidate Memories:
{candidates_string}

Select up to {top_k} memories that best support the answer.
If no memory is relevant, return an empty output.
If there are two or more memories, separate them with '###':"""


def episode_extraction_json_prompt(speaker1: str, speaker2: str, dialogues_text: str) -> str:
    return (
        episode_extraction_prompt(speaker1, speaker2, dialogues_text)
        + "\n\nReturn ONLY a JSON object with keys: p1, p2, t1, t2, shared, mutual.\n"
        "Each value must be an array of concise strings. No markdown, no extra keys."
    )


def episode_update_json_prompt(previous_data: EpisodeSessionData, current_data: EpisodeSessionData) -> str:
    return (
        episode_process_update_prompt(previous_data, current_data)
        + "\n\nReturn ONLY a JSON object with keys: p1, p2, t1, t2, shared, mutual.\n"
        "Each value must be an array of concise strings. No markdown, no extra keys."
    )


def episode_qa_selection_json_prompt(
    question: str,
    dia_no_tag_text: str,
    candidates: List[str],
    top_k: int,
) -> str:
    indexed = "\n".join(f"{idx}. {item}" for idx, item in enumerate(candidates))
    return f"""\
You are a memory retrieval analyst for question answering.
Select up to {top_k} candidate memories that directly support the answer.

Question:
{question}

Recent Dialogue:
{dia_no_tag_text}

Candidates:
{indexed}

Return ONLY JSON: {{"selected_indices": [int, ...]}}.
If no candidate is relevant, return an empty list."""


def episode_dialogue_selection_json_prompt(
    dia_no_tag_text: str,
    next_speaker: str,
    candidates: List[str],
    top_k: int,
) -> str:
    indexed = "\n".join(f"{idx}. {item}" for idx, item in enumerate(candidates))
    return f"""\
You are a conversation analyst.
Select up to {top_k} candidate memories most useful for the next part of the conversation by {next_speaker}.

Dialogue History:
{dia_no_tag_text}

Candidates:
{indexed}

Return ONLY JSON: {{"selected_indices": [int, ...]}}.
If no candidate is relevant, return an empty list."""


def parse_selected_indices_payload(payload: Dict, candidates: List[str], top_k: int) -> List[str]:
    raw = None
    for key in ("selected_indices", "indices", "selected"):
        if key in payload:
            raw = payload.get(key)
            break
    if not isinstance(raw, list):
        return []

    unique_indices: List[int] = []
    seen = set()
    for item in raw:
        idx: Optional[int] = None
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str):
            text = item.strip()
            if text.isdigit():
                idx = int(text)
            elif text in candidates:
                idx = candidates.index(text)
        elif isinstance(item, dict):
            maybe_idx = item.get("index")
            if isinstance(maybe_idx, int):
                idx = maybe_idx

        if idx is None:
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
        if len(unique_indices) >= top_k:
            break

    return [candidates[i] for i in unique_indices]


def episode_extract_selected_memories(text: str) -> List[str]:
    pattern = re.compile(r"separate them with '###':\s*(.*)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        raw = text.strip()
    else:
        raw = match.group(1).strip()
    if raw.lower().startswith("answer:"):
        raw = raw[len("answer:") :].strip()
    if not raw:
        return []
    if raw.lower() in {"none", "empty", "no relevant memory", "n/a"}:
        return []
    return [x.strip() for x in raw.split("###") if x.strip()]


def strict_extract_session_memory(
    llm: OpenAIJsonClient,
    dialogue_text: str,
    max_items: int,
) -> ShareMemory:
    session_data = EpisodeSessionData(
        s1_name="USER",
        s2_name="ASSISTANT",
        information=EpisodeSessionInformation(),
    )
    prompt = episode_extraction_prompt(session_data.s1_name, session_data.s2_name, dialogue_text)
    output = call_prompt_text(llm, prompt, retries=2)
    updated = episode_process_extraction(session_data, output)
    return memory_from_session_info(updated.information, max_items=max_items)


def strict_extract_session_memory_with_fallback(
    llm: OpenAIJsonClient,
    dialogue_text: str,
    max_items: int,
    json_retries: int,
    strict_json_io: bool,
) -> Tuple[ShareMemory, bool]:
    if strict_json_io:
        try:
            payload = llm.chat_json(
                [{"role": "user", "content": episode_extraction_json_prompt("USER", "ASSISTANT", dialogue_text)}],
                retries=json_retries,
            )
            return canonicalize_memory(
                normalize_memory(payload, max_items=max_items),
                max_items=max_items,
                retain_mutual=False,
            ), False
        except Exception:
            pass

    return strict_extract_session_memory(llm=llm, dialogue_text=dialogue_text, max_items=max_items), strict_json_io


def strict_update_memory(
    llm: OpenAIJsonClient,
    previous: ShareMemory,
    current: ShareMemory,
    max_items: int,
) -> ShareMemory:
    prev_data = EpisodeSessionData(
        s1_name="USER",
        s2_name="ASSISTANT",
        information=EpisodeSessionInformation(
            previous.p1,
            previous.p2,
            previous.t1,
            previous.t2,
            previous.shared,
            previous.mutual,
        ),
    )
    curr_data = EpisodeSessionData(
        s1_name="USER",
        s2_name="ASSISTANT",
        information=EpisodeSessionInformation(
            current.p1,
            current.p2,
            current.t1,
            current.t2,
            current.shared,
            current.mutual,
        ),
    )
    prompt = episode_process_update_prompt(prev_data, curr_data)
    output = call_prompt_text(llm, prompt, retries=2)
    normalized_output = episode_replace_names_with_uppercase(output, prev_data.s1_name, prev_data.s2_name)
    merged = episode_parse_memory(normalized_output, prev_data.s1_name, prev_data.s2_name)
    return memory_from_session_info(merged.information, max_items=max_items)


def strict_update_memory_with_fallback(
    llm: OpenAIJsonClient,
    previous: ShareMemory,
    current: ShareMemory,
    max_items: int,
    json_retries: int,
    strict_json_io: bool,
) -> Tuple[ShareMemory, bool]:
    if strict_json_io:
        try:
            prev_data = EpisodeSessionData(
                s1_name="USER",
                s2_name="ASSISTANT",
                information=EpisodeSessionInformation(
                    previous.p1,
                    previous.p2,
                    previous.t1,
                    previous.t2,
                    previous.shared,
                    previous.mutual,
                ),
            )
            curr_data = EpisodeSessionData(
                s1_name="USER",
                s2_name="ASSISTANT",
                information=EpisodeSessionInformation(
                    current.p1,
                    current.p2,
                    current.t1,
                    current.t2,
                    current.shared,
                    current.mutual,
                ),
            )
            payload = llm.chat_json(
                [{"role": "user", "content": episode_update_json_prompt(prev_data, curr_data)}],
                retries=json_retries,
            )
            return canonicalize_memory(
                normalize_memory(payload, max_items=max_items),
                max_items=max_items,
                retain_mutual=False,
            ), False
        except Exception:
            pass

    return strict_update_memory(
        llm=llm,
        previous=previous,
        current=current,
        max_items=max_items,
    ), strict_json_io


def strict_select_memories(
    llm: OpenAIJsonClient,
    question: str,
    memory: ShareMemory,
    recent_dialogue: str,
    top_k: int,
    include_mutual: bool,
    strict_selection_mode: str,
) -> List[str]:
    data = EpisodeSessionData(
        s1_name="USER",
        s2_name="ASSISTANT",
        information=EpisodeSessionInformation(
            memory.p1,
            memory.p2,
            memory.t1,
            memory.t2,
            memory.shared,
            memory.mutual,
        ),
    )
    candidates = flatten_memory_candidates_raw(memory, include_mutual=include_mutual)
    if strict_selection_mode == "dialogue":
        prompt = episode_selection_prompt(
            data=data,
            dia_no_tag_text=recent_dialogue,
            next_speaker="ASSISTANT",
            include_mutual=include_mutual,
        )
    else:
        prompt = episode_qa_selection_prompt(
            question=question,
            dia_no_tag_text=recent_dialogue,
            candidates=candidates,
            top_k=top_k,
        )
    output = call_prompt_text(llm, prompt, retries=2)
    raw_items = episode_extract_selected_memories(output)
    unique: List[str] = []
    seen = set()
    candidate_set = set(candidates)
    for item in raw_items:
        if item in candidate_set and item not in seen:
            seen.add(item)
            unique.append(item)
        if len(unique) >= top_k:
            break
    if unique:
        return unique
    return lexical_topk(
        question=question,
        candidates=flatten_memory_candidates_raw(memory, include_mutual=include_mutual),
        k=top_k,
    )


def strict_select_memories_with_fallback(
    llm: OpenAIJsonClient,
    question: str,
    memory: ShareMemory,
    recent_dialogue: str,
    top_k: int,
    include_mutual: bool,
    strict_selection_mode: str,
    json_retries: int,
    strict_json_io: bool,
) -> Tuple[List[str], bool]:
    candidates = flatten_memory_candidates_raw(memory, include_mutual=include_mutual)
    if not candidates:
        return [], False

    if strict_json_io:
        try:
            if strict_selection_mode == "dialogue":
                prompt = episode_dialogue_selection_json_prompt(
                    dia_no_tag_text=recent_dialogue,
                    next_speaker="ASSISTANT",
                    candidates=candidates,
                    top_k=top_k,
                )
            else:
                prompt = episode_qa_selection_json_prompt(
                    question=question,
                    dia_no_tag_text=recent_dialogue,
                    candidates=candidates,
                    top_k=top_k,
                )
            payload = llm.chat_json([{"role": "user", "content": prompt}], retries=json_retries)
            selected = parse_selected_indices_payload(payload, candidates, top_k)
            if selected:
                return selected, False
            return lexical_topk(question=question, candidates=candidates, k=top_k), False
        except Exception:
            pass

    return strict_select_memories(
        llm=llm,
        question=question,
        memory=memory,
        recent_dialogue=recent_dialogue,
        top_k=top_k,
        include_mutual=include_mutual,
        strict_selection_mode=strict_selection_mode,
    ), strict_json_io


def tail_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHARE-style memory pipeline on LongMemEval and export predictions JSONL."
    )
    parser.add_argument("--share-dir", type=Path, required=True, help="Path to SHARE repository root.")
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--memory-max-items",
        type=int,
        default=0,
        help="Per-bucket memory cap; 0 disables capping.",
    )
    parser.add_argument("--selection-top-k", type=int, default=6)
    parser.add_argument(
        "--strict-selection-mode",
        type=str,
        choices=("qa", "dialogue"),
        default="qa",
        help="Strict selection objective: 'qa' (LongMemEval-aligned) or 'dialogue' (original EPISODE next-turn style).",
    )
    parser.add_argument("--recent-turn-window", type=int, default=12)
    parser.add_argument(
        "--json-retries",
        type=int,
        default=1,
        help="Retries for JSON-only LLM steps (extract/update/select).",
    )
    parser.add_argument(
        "--max-session-dialogue-chars",
        type=int,
        default=8000,
        help="Clip each session dialogue to last N chars before extraction; 0 disables clipping.",
    )
    parser.add_argument(
        "--max-recent-dialogue-chars",
        type=int,
        default=6000,
        help="Clip recent dialogue snippet to last N chars before selection/answer; 0 disables clipping.",
    )
    parser.add_argument(
        "--session-progress-every",
        type=int,
        default=5,
        help="Update tqdm postfix every N ingested sessions.",
    )
    parser.add_argument(
        "--retain-mutual-memory",
        action="store_true",
        help="Keep mutual events as a separate memory bucket after update (off by default).",
    )
    parser.add_argument(
        "--include-mutual-in-candidates",
        action="store_true",
        help="Include mutual bucket in memory selection candidates (off by default).",
    )
    parser.add_argument(
        "--disable-strict-episode-memory",
        action="store_true",
        help="Disable official SHARE update_task prompt/parsing pipeline for memory extraction/update/selection.",
    )
    parser.add_argument(
        "--disable-strict-json-io",
        action="store_true",
        help="Disable JSON-constrained strict extraction/update/selection and use legacy text parsing only.",
    )
    parser.add_argument(
        "--force-abstain-when-uncertain",
        action="store_true",
        help="If set, enforce strict abstention behavior: output 'I don't know.' when uncertain.",
    )
    parser.add_argument("--omit-question-date", action="store_true")
    parser.add_argument("--preserve-session-order", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-override", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {path}, got {type(data)}")
    return data


def main() -> None:
    args = parse_args()

    if not args.share_dir.exists():
        raise FileNotFoundError(f"SHARE repo not found: {args.share_dir}")
    expected_share_path = args.share_dir / "evaluation" / "update_task"
    if not expected_share_path.exists():
        raise FileNotFoundError(f"Invalid SHARE repo layout (missing {expected_share_path})")

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
        llm = OpenAIJsonClient(
            api_key=api_key,
            model=args.llm_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    else:
        llm = None

    strict_episode_memory = not args.disable_strict_episode_memory
    strict_json_io = not args.disable_strict_json_io
    success = 0
    failed = 0
    started = time.time()

    with args.out_jsonl.open("w", encoding="utf-8") as pred_file:
        trace_file = args.trace_jsonl.open("w", encoding="utf-8") if args.trace_jsonl else None
        try:
            pbar = tqdm(dataset, total=len(dataset), desc="SHARE->LongMemEval", unit="q")
            for idx, entry in enumerate(pbar, start=1):
                qid = entry.get("question_id", f"idx_{idx}")
                qtype = entry.get("question_type", "unknown")
                try:
                    memory = ShareMemory()
                    history_pairs: List[Tuple[str, str]] = []
                    session_count = 0
                    strict_extract_fallbacks = 0
                    strict_update_fallbacks = 0
                    strict_extract_text_fallbacks = 0
                    strict_update_text_fallbacks = 0
                    strict_select_text_fallbacks = 0
                    ordered_sessions = get_ordered_sessions(entry, args.preserve_session_order)
                    total_sessions = len(ordered_sessions)
                    for _date_raw, turns in ordered_sessions:
                        pairs = list(iter_qa_pairs(turns))
                        if not pairs:
                            continue
                        dialogue_text = tail_chars(
                            format_pairs_as_dialogue(pairs),
                            args.max_session_dialogue_chars,
                        )
                        history_pairs.extend(pairs)
                        session_count += 1

                        if args.dry_run:
                            continue

                        if strict_episode_memory:
                            current_memory, used_text_fallback = strict_extract_session_memory_with_fallback(
                                llm=llm,
                                dialogue_text=dialogue_text,
                                max_items=args.memory_max_items,
                                json_retries=args.json_retries,
                                strict_json_io=strict_json_io,
                            )
                            if used_text_fallback:
                                strict_extract_text_fallbacks += 1
                            if memory_item_count(current_memory) == 0:
                                # Strict parser can be brittle with format drift; recover via JSON pipeline.
                                current_memory = extract_session_memory(
                                    llm=llm,
                                    dialogue_text=dialogue_text,
                                    max_items=args.memory_max_items,
                                    dry_run=False,
                                    json_retries=args.json_retries,
                                )
                                strict_extract_fallbacks += 1
                            previous_memory = memory
                            memory, used_text_fallback = strict_update_memory_with_fallback(
                                llm=llm,
                                previous=previous_memory,
                                current=current_memory,
                                max_items=args.memory_max_items,
                                json_retries=args.json_retries,
                                strict_json_io=strict_json_io,
                            )
                            if used_text_fallback:
                                strict_update_text_fallbacks += 1
                            if (
                                memory_item_count(memory) == 0
                                and memory_item_count(current_memory) > 0
                            ):
                                memory = update_share_memory(
                                    llm=llm,
                                    previous=previous_memory,
                                    current=current_memory,
                                    max_items=args.memory_max_items,
                                    dry_run=False,
                                    json_retries=args.json_retries,
                                    retain_mutual=args.retain_mutual_memory,
                                )
                                strict_update_fallbacks += 1
                        else:
                            current_memory = extract_session_memory(
                                llm=llm,
                                dialogue_text=dialogue_text,
                                max_items=args.memory_max_items,
                                dry_run=False,
                                json_retries=args.json_retries,
                            )
                            memory = update_share_memory(
                                llm=llm,
                                previous=memory,
                                current=current_memory,
                                max_items=args.memory_max_items,
                                dry_run=False,
                                json_retries=args.json_retries,
                                retain_mutual=args.retain_mutual_memory,
                            )
                        step = max(1, args.session_progress_every)
                        if session_count % step == 0 or session_count == total_sessions:
                            elapsed = time.time() - started
                            pbar.set_postfix(
                                ok=success,
                                fail=failed,
                                last=qid,
                                stage="ingest",
                                sess=f"{session_count}/{total_sessions}",
                                elapsed_s=f"{elapsed:.1f}",
                            )

                    question = (entry.get("question") or "").strip()
                    query_with_date = question
                    if not args.omit_question_date and entry.get("question_date"):
                        query_with_date = f"Current date: {entry['question_date']}\n\n{question}"

                    recent_pairs = history_pairs[-args.recent_turn_window :] if args.recent_turn_window > 0 else history_pairs
                    recent_dialogue = tail_chars(
                        format_pairs_as_dialogue(recent_pairs),
                        args.max_recent_dialogue_chars,
                    )
                    if strict_episode_memory:
                        candidates = flatten_memory_candidates_raw(
                            memory,
                            include_mutual=args.include_mutual_in_candidates,
                        )
                        if args.dry_run:
                            selected = lexical_topk(question, candidates, args.selection_top_k)
                        else:
                            selected, used_text_fallback = strict_select_memories_with_fallback(
                                llm=llm,
                                question=question,
                                memory=memory,
                                recent_dialogue=recent_dialogue,
                                top_k=args.selection_top_k,
                                include_mutual=args.include_mutual_in_candidates,
                                strict_selection_mode=args.strict_selection_mode,
                                json_retries=args.json_retries,
                                strict_json_io=strict_json_io,
                            )
                            if used_text_fallback:
                                strict_select_text_fallbacks += 1
                    else:
                        candidates = flatten_memory_candidates(
                            memory,
                            include_mutual=args.include_mutual_in_candidates,
                        )
                        selected = select_memories(
                            llm=llm,
                            question=question,
                            recent_dialogue=recent_dialogue,
                            candidates=candidates,
                            top_k=args.selection_top_k,
                            dry_run=args.dry_run,
                            json_retries=args.json_retries,
                        )
                    hypothesis = answer_question(
                        llm=llm,
                        question=question,
                        query_with_date=query_with_date,
                        selected_memories=selected,
                        recent_dialogue=recent_dialogue,
                        dry_run=args.dry_run,
                        force_abstain_when_uncertain=args.force_abstain_when_uncertain,
                    )

                    pred_obj = {"question_id": qid, "hypothesis": hypothesis}
                    pred_file.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
                    pred_file.flush()

                    if trace_file:
                        trace_obj = {
                            "question_id": qid,
                            "question_type": qtype,
                            "strict_episode_memory": strict_episode_memory,
                            "strict_selection_mode": args.strict_selection_mode if strict_episode_memory else "json",
                            "strict_json_io": strict_json_io if strict_episode_memory else False,
                            "n_sessions": session_count,
                            "n_pairs": len(history_pairs),
                            "memory": asdict(memory),
                            "n_candidates": len(candidates),
                            "selected_memories": selected,
                            "memory_item_count": memory_item_count(memory),
                            "strict_extract_fallbacks": strict_extract_fallbacks,
                            "strict_update_fallbacks": strict_update_fallbacks,
                            "strict_extract_text_fallbacks": strict_extract_text_fallbacks,
                            "strict_update_text_fallbacks": strict_update_text_fallbacks,
                            "strict_select_text_fallbacks": strict_select_text_fallbacks,
                            "query_used": query_with_date,
                        }
                        trace_file.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")
                        trace_file.flush()

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
