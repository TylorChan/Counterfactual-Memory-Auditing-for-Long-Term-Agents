# Cross-Machine Handoff Log

This file is the shared working memory across machines.

How to use:
- Before switching machines, run `scripts/handoff.sh --machine "<machine-name>"`.
- The script auto-fills summary/progress/resolved/pending from local git state and run artifacts.
- Fill in both:
- Execution state: `What changed`, `Run status`, `Blockers`, `Next 3 steps`.
- Conversation state: `Conversation summary`, `Resolved issues`, `Current progress`, `Pending decisions`.
- Commit and push this file with your code changes.
- On the next machine, pull latest and read the newest handoff entry first.

Recommended machine names:
- `Google-VM`
- `MSI`
- `MacBook-Pro16`
