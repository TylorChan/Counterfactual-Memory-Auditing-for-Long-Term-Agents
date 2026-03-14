#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_locator import VisionLocator


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def startup_countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    log(
        f"Startup countdown: {seconds}s. Put the mouse where you want and keep the ChatGPT app ready."
    )
    for remaining in range(seconds, 0, -1):
        log(f"Automation starts in {remaining}s")
        time.sleep(1)


def run_checked(cmd: list[str], input_text: str | None = None) -> str:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def set_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True, check=True)


def get_clipboard() -> str:
    return run_checked(["pbpaste"]).rstrip("\n")


def activate_app(app_name: str) -> None:
    script = f'tell application "{app_name}" to activate'
    run_checked(["osascript", "-e", script])


def keystroke(key: str, modifiers: list[str] | None = None) -> None:
    modifiers = modifiers or []
    if modifiers:
        mods = ", ".join(f"{item} down" for item in modifiers)
        script = (
            'tell application "System Events" to '
            f'keystroke "{key}" using {{{mods}}}'
        )
    else:
        script = f'tell application "System Events" to keystroke "{key}"'
    run_checked(["osascript", "-e", script])


def key_code(code: int, modifiers: list[str] | None = None) -> None:
    modifiers = modifiers or []
    if modifiers:
        mods = ", ".join(f"{item} down" for item in modifiers)
        script = (
            'tell application "System Events" to '
            f'key code {code} using {{{mods}}}'
        )
    else:
        script = f'tell application "System Events" to key code {code}'
    run_checked(["osascript", "-e", script])


def screencap_hash(region: dict[str, int]) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        rect = f"{region['x']},{region['y']},{region['width']},{region['height']}"
        subprocess.run(
            ["screencapture", "-x", "-R", rect, str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass
class UiPoint:
    x: int
    y: int


class MacChatGPTExecutor:
    def __init__(
        self,
        profile: dict[str, Any],
        trace_dir: Path | None = None,
        vision_locator: VisionLocator | None = None,
    ) -> None:
        self.profile = profile
        self.trace_dir = trace_dir
        self.vision_locator = vision_locator
        self.click_pause = float(profile.get("click_pause_seconds", 0.35))
        self.post_paste_pause = float(profile.get("post_paste_pause_seconds", 0.2))
        self.stability_timeout = float(profile.get("stability_timeout_seconds", 120.0))
        self.stability_required = float(profile.get("stability_required_seconds", 2.5))
        self.stability_poll = float(profile.get("stability_poll_seconds", 0.6))
        self.vision_confidence_threshold = float(
            profile.get("vision_confidence_threshold", 0.6)
        )
        self.dynamic_targets = {
            "project_home_button",
            "project_home_composer_input",
            "copy_last_response_button",
        }

    def point(self, *names: str) -> UiPoint:
        for name in names:
            item = self.profile.get(name)
            if item is not None:
                return UiPoint(int(item["x"]), int(item["y"]))
        joined = ", ".join(names)
        raise KeyError(f"None of the UI points are present in the profile: {joined}")

    def point_name(self, *names: str) -> str:
        for name in names:
            if self.vision_locator and name in self.dynamic_targets:
                return name
            if name in self.profile:
                return name
        joined = ", ".join(names)
        raise KeyError(f"None of the UI points are present in the profile: {joined}")

    def click_any(self, *names: str) -> None:
        chosen_name = self.point_name(*names)
        if self.vision_locator and chosen_name in self.dynamic_targets:
            located = self.vision_locator.locate(chosen_name)
            if (
                located.confidence >= self.vision_confidence_threshold
                or chosen_name not in self.profile
            ):
                point = UiPoint(located.x, located.y)
                log(
                    f"Vision located `{chosen_name}` at ({point.x}, {point.y}) "
                    f"confidence={located.confidence:.2f} screenshot={located.screenshot_path}"
                )
            else:
                point = self.point(chosen_name)
                log(
                    f"Vision confidence too low for `{chosen_name}` "
                    f"({located.confidence:.2f} < {self.vision_confidence_threshold:.2f}); "
                    f"fallback to calibrated point ({point.x}, {point.y})"
                )
        else:
            point = self.point(chosen_name)
            log(f"Click `{chosen_name}` at ({point.x}, {point.y})")
        script = (
            'tell application "System Events" to '
            f'click at {{{point.x}, {point.y}}}'
        )
        run_checked(["osascript", "-e", script])
        time.sleep(self.click_pause)

    def click(self, name: str) -> None:
        self.click_any(name)

    def focus_app(self) -> None:
        log(f"Activate app `{self.profile.get('app_name', 'ChatGPT')}`")
        activate_app(str(self.profile.get("app_name", "ChatGPT")))
        time.sleep(0.8)

    def paste_text(self, text: str, submit: bool = True) -> None:
        preview = text.replace("\n", " ")[:100]
        log(f"Paste text: {preview!r}")
        set_clipboard(text)
        keystroke("v", ["command"])
        time.sleep(self.post_paste_pause)
        if submit:
            log("Submit message")
            key_code(36)

    def wait_for_stability(self) -> None:
        region = self.profile["stability_region"]
        log(
            "Wait for UI stability in region "
            f"({region['x']}, {region['y']}, {region['width']}, {region['height']})"
        )
        start = time.time()
        stable_since: float | None = None
        last_hash = None
        while True:
            current_hash = screencap_hash(region)
            now = time.time()
            if current_hash == last_hash:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= self.stability_required:
                    log("UI stabilized")
                    return
            else:
                last_hash = current_hash
                stable_since = None
            if now - start > self.stability_timeout:
                raise TimeoutError("UI did not stabilize before timeout.")
            time.sleep(self.stability_poll)

    def create_project(self, project_name: str) -> None:
        log(f"Create project `{project_name}`")
        self.focus_app()
        self.click("create_project_button")
        self.click("project_name_input")
        self.paste_text(project_name, submit=False)
        self.click("more_options_button")
        self.click("project_only_option")
        self.click("project_settings_done_button")
        self.click("project_create_confirm_button")
        time.sleep(1.2)
        self.verify_project_title(project_name)

    def open_new_session(self) -> None:
        log("Navigate to the project home so the next send starts a new session")
        self.focus_app()
        self.click_any("project_home_button", "new_chat_button")
        time.sleep(1.0)

    def send_message(self, text: str, composer_name: str = "composer_input") -> None:
        self.focus_app()
        self.click_any(composer_name)
        self.paste_text(text, submit=True)
        self.wait_for_stability()

    def session_start_composer_name(self) -> str:
        if self.vision_locator:
            return "project_home_composer_input"
        if "project_home_composer_input" in self.profile:
            return "project_home_composer_input"
        return "composer_input"

    def copy_last_answer(self) -> str:
        log("Copy the last assistant answer")
        self.focus_app()
        self.click("copy_last_response_button")
        time.sleep(0.4)
        return get_clipboard().strip()

    def verify_project_title(self, expected_title: str) -> None:
        log(f"Verify current project title is `{expected_title}`")
        if self.vision_locator and "project_title_anchor" in self.profile:
            matched = self.vision_locator.project_title_matches(
                expected_title,
                self.profile["project_title_anchor"],
            )
            if not matched:
                raise RuntimeError(
                    f"Project title verification failed. Expected current project `{expected_title}`."
                )
            log("Project title verification passed")
            return
        log("No vision title verifier available; skip project title verification")


def normalize_session_text(session_date: str, session_turns: list[dict[str, str]]) -> list[str]:
    messages = [f"Date: {session_date}"]
    messages.extend(turn["content"] for turn in session_turns if turn["role"] == "user")
    return messages


def load_sample(dataset_path: Path, question_id: str) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for item in data:
        if item["question_id"] == question_id:
            return item
    raise KeyError(f"question_id not found: {question_id}")


def make_final_query(sample: dict[str, Any]) -> str:
    return f"Current date: {sample['question_date']}\n\n{sample['question']}"


def run_question(
    executor: MacChatGPTExecutor,
    sample: dict[str, Any],
    out_path: Path,
    trace_path: Path,
    project_prefix: str,
    max_sessions: int | None,
) -> None:
    question_id = sample["question_id"]
    project_name = f"{project_prefix}_{question_id}"
    log(
        f"Start question_id={question_id} "
        f"type={sample['question_type']} project={project_name}"
    )
    trace: dict[str, Any] = {
        "question_id": question_id,
        "question_type": sample["question_type"],
        "project_name": project_name,
        "events": [],
    }

    executor.create_project(project_name)
    trace["events"].append({"event": "create_project"})

    session_iter = zip(
        sample["haystack_dates"],
        sample["haystack_session_ids"],
        sample["haystack_sessions"],
    )
    if max_sessions is not None:
        session_iter = list(session_iter)[:max_sessions]

    session_list = list(session_iter)
    total_sessions = len(session_list)
    total_user_messages = sum(
        1 + sum(1 for turn in turns if turn["role"] == "user")
        for _, _, turns in session_list
    )
    log(
        f"Replay {total_sessions} sessions with {total_user_messages} user-side messages "
        "(including one date message per session)"
    )

    for session_idx, (session_date, session_id, session_turns) in enumerate(session_list, start=1):
        user_messages = normalize_session_text(session_date, session_turns)
        log(
            f"Session {session_idx}/{total_sessions}: "
            f"session_id={session_id} date={session_date} messages={len(user_messages)}"
        )
        executor.open_new_session()
        trace["events"].append(
            {
                "event": "open_session",
                "session_idx": session_idx - 1,
                "session_id": session_id,
                "session_date": session_date,
            }
        )
        for msg_idx, message in enumerate(user_messages, start=1):
            log(f"  Send message {msg_idx}/{len(user_messages)} for session {session_idx}")
            composer_name = (
                executor.session_start_composer_name()
                if msg_idx == 1
                else "composer_input"
            )
            executor.send_message(message, composer_name=composer_name)
            trace["events"].append(
                {
                    "event": "send_message",
                    "session_idx": session_idx - 1,
                    "message_idx": msg_idx - 1,
                    "preview": message[:120],
                }
            )

    log("Open final query session")
    executor.open_new_session()
    trace["events"].append({"event": "open_final_query_session"})
    final_query = make_final_query(sample)
    log("Send final query")
    executor.send_message(final_query, composer_name=executor.session_start_composer_name())
    trace["events"].append({"event": "send_final_query", "preview": final_query[:120]})
    answer = executor.copy_last_answer()
    log(f"Captured final answer preview: {answer[:160]!r}")
    trace["events"].append({"event": "copy_last_answer", "answer_preview": answer[:200]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": question_id, "hypothesis": answer}, ensure_ascii=False) + "\n")
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote prediction to {out_path}")
    log(f"Wrote trace to {trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LongMemEval question through the ChatGPT macOS app.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/daqingchen/csci8980/LongMemEval/data/longmemeval_s_cleaned_50.json"),
    )
    parser.add_argument("--question-id", required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("profile.local.json"),
        help="Calibrated UI profile JSON.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("/Users/daqingchen/csci8980/LongMemEval/preds_chatgpt_project_memory_smoke.jsonl"),
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=Path("/Users/daqingchen/csci8980/chatgpt_project_eval/traces/last_run_trace.json"),
    )
    parser.add_argument("--project-prefix", default="lme")
    parser.add_argument(
        "--locator",
        choices=["fixed", "vision"],
        default="fixed",
        help="Use fixed calibrated points or a vision model for selected dynamic targets.",
    )
    parser.add_argument(
        "--locator-model",
        default="gpt-4.1-mini",
        help="Vision-capable OpenAI model used when --locator vision.",
    )
    parser.add_argument(
        "--vision-confidence-threshold",
        type=float,
        default=None,
        help="Optional override for the minimum confidence required before using a vision-located point.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional API key override for the vision locator.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI base URL override for the vision locator.",
    )
    parser.add_argument(
        "--startup-countdown-seconds",
        type=int,
        default=5,
        help="Seconds to wait before the automation starts.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional engineering smoke limit. When set, only the first N haystack sessions are replayed.",
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if args.vision_confidence_threshold is not None:
        profile["vision_confidence_threshold"] = args.vision_confidence_threshold
    sample = load_sample(args.dataset, args.question_id)
    startup_countdown(args.startup_countdown_seconds)
    vision_locator = None
    if args.locator == "vision":
        vision_locator = VisionLocator(
            model=args.locator_model,
            trace_dir=args.trace_json.parent / "locator",
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
        )
        log(f"Use vision locator with model `{args.locator_model}`")
    executor = MacChatGPTExecutor(
        profile=profile,
        trace_dir=args.trace_json.parent,
        vision_locator=vision_locator,
    )
    run_question(
        executor=executor,
        sample=sample,
        out_path=args.out_jsonl,
        trace_path=args.trace_json,
        project_prefix=args.project_prefix,
        max_sessions=args.max_sessions,
    )
    log(f"Finished question_id={args.question_id}")


if __name__ == "__main__":
    main()
