#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


TARGET_DESCRIPTIONS = {
    "create_project_button": (
        "Locate the `New project` button in the left sidebar of the ChatGPT macOS app."
    ),
    "project_home_button": (
        "Locate the UI control in the ChatGPT macOS app that returns the current "
        "chat view to the enclosing project home or project overview."
    ),
    "project_home_composer_input": (
        "Locate the text input composer on the project home / project overview page. "
        "Sending a message from this composer starts a new session inside the project."
    ),
    "composer_input": (
        "Locate the main text input composer at the bottom of the active chat page."
    ),
    "copy_last_response_button": (
        "Locate the copy button associated with the most recent assistant response "
        "in the conversation area."
    ),
}


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def screenshot_fullscreen(out_path: Path) -> None:
    subprocess.run(
        ["screencapture", "-x", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def image_to_data_url(path: Path) -> str:
    mime = "image/png"
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{raw}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model output: {text[:400]}")
    return json.loads(match.group(0))


@dataclass
class LocatedPoint:
    x: int
    y: int
    confidence: float
    screenshot_path: Path
    raw_response: str


class VisionLocator:
    def __init__(
        self,
        model: str,
        trace_dir: Path,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        self.model = model
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv(Path("/Users/daqingchen/csci8980/.env"))
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for vision locator.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if openai_base_url or os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**kwargs)

    def locate(self, target_name: str) -> LocatedPoint:
        description = TARGET_DESCRIPTIONS[target_name]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.trace_dir / f"{timestamp}_{target_name}.png"
        screenshot_fullscreen(screenshot_path)
        with Image.open(screenshot_path) as image:
            width, height = image.size
        data_url = image_to_data_url(screenshot_path)
        prompt = (
            "You are a UI element locator for a macOS desktop screenshot.\n"
            f"Image size: width={width}, height={height}.\n"
            f"Task: {description}\n"
            "Return ONLY valid JSON with this schema:\n"
            '{"x": <int>, "y": <int>, "confidence": <float 0-1>, "reason": "<short text>"}\n'
            "Coordinates must be absolute screen pixel coordinates in this screenshot.\n"
            "Choose a clickable point centered on the target element.\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        payload = extract_json_object(raw)
        return LocatedPoint(
            x=int(payload["x"]),
            y=int(payload["y"]),
            confidence=float(payload.get("confidence", 0.0)),
            screenshot_path=screenshot_path,
            raw_response=raw,
        )

    def project_title_matches(self, expected_title: str, anchor: dict[str, int]) -> bool:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.trace_dir / f"{timestamp}_project_title_check.png"
        screenshot_fullscreen(screenshot_path)
        with Image.open(screenshot_path) as image:
            width, height = image.size
        data_url = image_to_data_url(screenshot_path)
        prompt = (
            "You are validating the currently open ChatGPT project title from a macOS screenshot.\n"
            f"Image size: width={width}, height={height}.\n"
            f"Expected project title: {expected_title}\n"
            f"A calibrated anchor near the project title is approximately at x={anchor['x']}, y={anchor['y']}.\n"
            "Return ONLY valid JSON with schema:\n"
            '{"matches": <true|false>, "observed_title": "<string>", "confidence": <float 0-1>}\n'
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        payload = extract_json_object(raw)
        return bool(payload.get("matches", False))
