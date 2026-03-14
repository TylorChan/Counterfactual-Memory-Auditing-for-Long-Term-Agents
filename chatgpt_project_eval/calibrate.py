#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyautogui


POINT_FIELDS = [
    "create_project_button",
    "project_name_input",
    "more_options_button",
    "project_only_option",
    "project_settings_done_button",
    "project_create_confirm_button",
    "project_title_anchor",
    "project_home_button",
    "project_home_composer_input",
    "composer_input",
    "copy_last_response_button",
]

RECT_FIELDS = ["stability_region"]


def wait_for_hover(message: str, countdown_seconds: int) -> None:
    print(message, flush=True)
    input(f"Press Enter to start a {countdown_seconds}-second countdown...")
    for remaining in range(countdown_seconds, 0, -1):
        print(f"  Capture in {remaining}...", flush=True)
        time.sleep(1)


def capture_point(name: str, countdown_seconds: int) -> dict[str, int]:
    wait_for_hover(f"Hover the mouse over `{name}`.", countdown_seconds)
    x, y = pyautogui.position()
    print(f"Captured `{name}` at ({x}, {y})", flush=True)
    return {"x": int(x), "y": int(y)}


def capture_rect(name: str, countdown_seconds: int) -> dict[str, int]:
    wait_for_hover(
        f"Hover over the TOP-LEFT corner of `{name}`.",
        countdown_seconds,
    )
    x1, y1 = pyautogui.position()
    print(f"Top-left at ({x1}, {y1})", flush=True)
    time.sleep(0.2)
    wait_for_hover(
        f"Hover over the BOTTOM-RIGHT corner of `{name}`.",
        countdown_seconds,
    )
    x2, y2 = pyautogui.position()
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width == 0 or height == 0:
        raise SystemExit(f"{name} must have non-zero width and height.")
    print(
        f"Captured `{name}` at x={left}, y={top}, width={width}, height={height}",
        flush=True,
    )
    return {"x": int(left), "y": int(top), "width": int(width), "height": int(height)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ChatGPT app UI coordinates.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("profile.local.json"),
        help="Output profile JSON path.",
    )
    parser.add_argument(
        "--app-name",
        default="ChatGPT",
        help="macOS app name to activate before running automation.",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=int,
        default=5,
        help="Seconds to move the mouse before each capture.",
    )
    args = parser.parse_args()

    profile: dict[str, object] = {
        "app_name": args.app_name,
        "click_pause_seconds": 0.35,
        "post_paste_pause_seconds": 0.2,
        "stability_timeout_seconds": 120.0,
        "stability_required_seconds": 2.5,
        "stability_poll_seconds": 0.6,
    }

    print(
        "Keep the ChatGPT app visible in a fixed position before capturing coordinates.",
        flush=True,
    )
    print(
        "This profile is machine- and layout-specific. Recalibrate if the window moves or the UI changes.",
        flush=True,
    )

    for field in POINT_FIELDS:
        profile[field] = capture_point(field, args.countdown_seconds)

    for field in RECT_FIELDS:
        profile[field] = capture_rect(field, args.countdown_seconds)

    args.out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Saved profile to {args.out}", flush=True)


if __name__ == "__main__":
    main()
