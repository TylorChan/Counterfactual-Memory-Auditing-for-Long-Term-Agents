#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_one import MacChatGPTExecutor, log, startup_countdown
from vision_locator import VisionLocator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug only the create-project click path in the ChatGPT macOS app."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("profile.local.json"),
        help="Calibrated UI profile JSON.",
    )
    parser.add_argument(
        "--startup-countdown-seconds",
        type=int,
        default=5,
        help="Seconds to wait before the automation starts.",
    )
    parser.add_argument(
        "--pause-after-click-seconds",
        type=int,
        default=3,
        help="Seconds to wait after clicking New project so you can inspect the UI.",
    )
    parser.add_argument(
        "--locator",
        choices=["fixed", "vision"],
        default="fixed",
        help="Use fixed calibrated point or vision locator for the New project click.",
    )
    parser.add_argument(
        "--locator-model",
        default="gpt-5.4",
        help="Vision-capable OpenAI model used when --locator vision.",
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    startup_countdown(args.startup_countdown_seconds)
    vision_locator = None
    if args.locator == "vision":
        vision_locator = VisionLocator(
            model=args.locator_model,
            trace_dir=Path("/Users/daqingchen/csci8980/chatgpt_project_eval/traces/locator"),
        )
        located = vision_locator.locate("create_project_button")
        log(
            f"Vision located create_project_button at ({located.x}, {located.y}) "
            f"confidence={located.confidence:.2f} screenshot={located.screenshot_path}"
        )
        profile["create_project_button"] = {"x": located.x, "y": located.y}
    executor = MacChatGPTExecutor(profile=profile, vision_locator=None)

    log("Debug create-project step only")
    executor.focus_app()
    executor.click("create_project_button")
    log(
        f"Clicked create_project_button. Waiting {args.pause_after_click_seconds}s for inspection."
    )
    time.sleep(args.pause_after_click_seconds)
    log("Debug click finished")


if __name__ == "__main__":
    main()
