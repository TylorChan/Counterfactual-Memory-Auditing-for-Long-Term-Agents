# ChatGPT Project Memory Eval

Deterministic macOS UI automation for testing ChatGPT app project-only memory on LongMemEval samples.

This is intentionally not a free-form agent controller. It uses:

- a calibrated UI profile
- fixed click targets
- clipboard paste for all inputs
- screenshot-based UI stabilization before the next step

## Install

```bash
cd /Users/daqingchen/csci8980
python3 -m venv .venv-chatgpt-eval
source .venv-chatgpt-eval/bin/activate
pip install -r /Users/daqingchen/csci8980/chatgpt_project_eval/requirements.txt
```

Grant macOS Accessibility and Screen Recording permissions to:

- Terminal or iTerm
- Python

## Calibrate

Open the ChatGPT macOS app and keep it in a fixed position.

```bash
source /Users/daqingchen/csci8980/.venv-chatgpt-eval/bin/activate
python /Users/daqingchen/csci8980/chatgpt_project_eval/calibrate.py \
  --out /Users/daqingchen/csci8980/chatgpt_project_eval/profile.local.json
```

The generated profile is layout-specific. Re-run calibration if the app window moves or the UI changes.
Each capture uses a countdown, so you can press Enter in the terminal and then move the mouse to the target UI element.

## Smoke test one question

Before running:

- disable web search in ChatGPT
- ensure the account can create projects
- verify the app is logged in

```bash
source /Users/daqingchen/csci8980/.venv-chatgpt-eval/bin/activate
python /Users/daqingchen/csci8980/chatgpt_project_eval/run_one.py \
  --question-id gpt4_9a159967 \
  --max-sessions 2 \
  --profile /Users/daqingchen/csci8980/chatgpt_project_eval/profile.local.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_chatgpt_project_memory_smoke.jsonl \
  --trace-json /Users/daqingchen/csci8980/chatgpt_project_eval/traces/gpt4_9a159967.json
```

Use `--max-sessions` only for automation smoke tests. Remove it for a real LongMemEval run.
The runner prints timestamped progress logs for project creation, per-session replay, UI stabilization, and final answer capture.
It also includes a startup countdown so you can position the mouse and verify the app state before automation begins.

## Debug the New project click

Use this before a full run if you suspect the `New project` coordinate is wrong.

```bash
source /Users/daqingchen/csci8980/.venv-chatgpt-eval/bin/activate
python /Users/daqingchen/csci8980/chatgpt_project_eval/debug_create_project.py \
  --profile /Users/daqingchen/csci8980/chatgpt_project_eval/profile.local.json \
  --startup-countdown-seconds 5 \
  --pause-after-click-seconds 3
```

To test only GPT-5.4 screenshot-based localization for `New project`:

```bash
source /Users/daqingchen/csci8980/.venv-chatgpt-eval/bin/activate
python /Users/daqingchen/csci8980/chatgpt_project_eval/debug_create_project.py \
  --locator vision \
  --locator-model gpt-5.4 \
  --profile /Users/daqingchen/csci8980/chatgpt_project_eval/profile.local.json \
  --startup-countdown-seconds 5 \
  --pause-after-click-seconds 3
```

## Optional vision locator

If dynamic UI elements move too much for fixed coordinates, use a vision model only for selected targets such as:

- `create_project_button`
- `project_name_input`
- `more_options_button`
- `project_only_option`
- `project_settings_done_button`
- `project_create_confirm_button`
- `project_home_button`
- `project_home_composer_input`
- `copy_last_response_button`

The main flow remains deterministic; the model only returns click coordinates from a screenshot.

Example:

```bash
source /Users/daqingchen/csci8980/.venv-chatgpt-eval/bin/activate
python /Users/daqingchen/csci8980/chatgpt_project_eval/run_one.py \
  --question-id 852ce960 \
  --max-sessions 2 \
  --locator vision \
  --locator-model gpt-4.1-mini \
  --vision-confidence-threshold 0.6 \
  --profile /Users/daqingchen/csci8980/chatgpt_project_eval/profile.local.json \
  --out-jsonl /Users/daqingchen/csci8980/LongMemEval/preds_chatgpt_project_memory_smoke.jsonl \
  --trace-json /Users/daqingchen/csci8980/chatgpt_project_eval/traces/852ce960.smoke.json
```

If a vision-located point has confidence below the threshold and the profile contains a calibrated fallback point, the runner uses the calibrated point instead.

## Protocol implemented

For one `question_id`, the runner:

1. creates a new project
2. switches it to `Project-only memory`
3. navigates back to the project home before each session
4. starts one new session by sending the first message from the project-home composer
5. sends `Date: <session_date>` plus each user turn
6. starts one final session in the same project
7. asks the final question with `Current date: <question_date>`
8. copies the last assistant response into a LongMemEval-style JSONL row

## Current limitations

- The automation is calibrated by screen coordinates, not semantic UI parsing.
- The create-project modal currently relies on calibrated fixed points. Recalibrate carefully in the exact final window layout.
- `project_title_anchor` should point near the visible project title so the vision verifier can confirm the correct project was opened.
- `project_home_button` should point to the control that returns you to the project home.
- `project_home_composer_input` should point to the composer on the project home page; this may differ from the normal chat composer.
- `copy_last_response_button` must point to the last assistant message's copy button in your current layout.
- The stability detector watches a fixed screenshot region. Calibrate it to the main conversation pane.
- This script is intended for controlled single-machine experiments, not arbitrary desktop automation.
