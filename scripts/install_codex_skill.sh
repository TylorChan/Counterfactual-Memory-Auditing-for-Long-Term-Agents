#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <skill-name> [repo-root]" >&2
  exit 1
fi

SKILL_NAME="$1"
REPO_ROOT="${2:-$(git rev-parse --show-toplevel)}"
SRC_DIR="$REPO_ROOT/codex_skills/$SKILL_NAME"
DEST_ROOT="${CODEX_HOME:-$HOME/.codex}/skills"
DEST_DIR="$DEST_ROOT/$SKILL_NAME"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Skill not found in repo: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"
rm -rf "$DEST_DIR"
cp -R "$SRC_DIR" "$DEST_DIR"

echo "Installed skill to $DEST_DIR"
echo "Restart Codex to pick up new skills."
