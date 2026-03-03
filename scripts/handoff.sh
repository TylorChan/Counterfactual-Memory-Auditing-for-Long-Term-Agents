#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/handoff.sh [--machine NAME] [--file PATH] [--note TEXT] [--conv-summary TEXT] [--progress TEXT] [--resolved TEXT] [--pending TEXT] [--dry-run]

Examples:
  scripts/handoff.sh --machine "Google-VM"
  scripts/handoff.sh --machine "MacBook-Pro16"
  scripts/handoff.sh --machine "Google-VM" --note "Before switching to MSI"
  scripts/handoff.sh --dry-run
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_FILE="${REPO_ROOT}/ops/HANDOFF.md"
DEFAULT_MACHINE="${HOSTNAME:-$(hostname -s 2>/dev/null || echo unknown-machine)}"

MACHINE="${DEFAULT_MACHINE}"
OUT_FILE="${DEFAULT_FILE}"
NOTE=""
CONV_SUMMARY=""
PROGRESS_TEXT=""
RESOLVED_TEXT=""
PENDING_TEXT=""
DRY_RUN=0

join_with_semicolon() {
  local out=""
  local item=""
  for item in "$@"; do
    if [[ -z "${item}" ]]; then
      continue
    fi
    if [[ -z "${out}" ]]; then
      out="${item}"
    else
      out="${out}; ${item}"
    fi
  done
  echo "${out}"
}

trim_one_line() {
  echo "$1" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

has_changed() {
  local pattern="$1"
  printf '%s\n' "${CHANGED_FILES}" | grep -Eq "${pattern}"
}

progress_from_preds() {
  local label="$1"
  local file="$2"
  local total="$3"
  if [[ -f "${file}" ]]; then
    local n
    n="$(wc -l < "${file}" | tr -d ' ')"
    if [[ "${n}" -gt "${total}" ]]; then
      n="${total}"
    fi
    echo "${label} ${n}/${total} ($(basename "${file}"))"
  else
    echo "${label} not started"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine)
      [[ $# -ge 2 ]] || { echo "Error: --machine requires a value."; exit 1; }
      MACHINE="$2"
      shift 2
      ;;
    --file)
      [[ $# -ge 2 ]] || { echo "Error: --file requires a value."; exit 1; }
      OUT_FILE="$2"
      shift 2
      ;;
    --note)
      [[ $# -ge 2 ]] || { echo "Error: --note requires a value."; exit 1; }
      NOTE="$2"
      shift 2
      ;;
    --conv-summary)
      [[ $# -ge 2 ]] || { echo "Error: --conv-summary requires a value."; exit 1; }
      CONV_SUMMARY="$2"
      shift 2
      ;;
    --progress)
      [[ $# -ge 2 ]] || { echo "Error: --progress requires a value."; exit 1; }
      PROGRESS_TEXT="$2"
      shift 2
      ;;
    --resolved)
      [[ $# -ge 2 ]] || { echo "Error: --resolved requires a value."; exit 1; }
      RESOLVED_TEXT="$2"
      shift 2
      ;;
    --pending)
      [[ $# -ge 2 ]] || { echo "Error: --pending requires a value."; exit 1; }
      PENDING_TEXT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'."
      usage
      exit 1
      ;;
  esac
done

UTC_NOW="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown-branch)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown-commit)"
STATUS_PORCELAIN="$(git -C "${REPO_ROOT}" status --short 2>/dev/null || true)"
STATUS_LINES="${STATUS_PORCELAIN}"
if [[ -z "${STATUS_LINES}" ]]; then
  STATUS_LINES="(clean)"
fi
CHANGED_FILES="$(printf '%s\n' "${STATUS_PORCELAIN}" | sed -E 's/^.. //' | sed '/^$/d')"
CHANGED_COUNT="$(printf '%s\n' "${CHANGED_FILES}" | sed '/^$/d' | wc -l | tr -d ' ')"
TOP_AREAS="$(printf '%s\n' "${CHANGED_FILES}" | sed 's/ -> /\n/g' | awk -F/ 'NF{print $1}' | sed '/^$/d' | sort -u | head -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/, /g')"
if [[ -z "${TOP_AREAS}" ]]; then
  TOP_AREAS="(none)"
fi

AUTO_NOTE="Auto handoff snapshot from git diff/status."
if [[ "${CHANGED_COUNT}" -eq 0 ]]; then
  AUTO_NOTE="Auto handoff snapshot: working tree clean."
fi

declare -a auto_conv_parts
if [[ "${CHANGED_COUNT}" -gt 0 ]]; then
  auto_conv_parts+=("Auto-generated from git changes (${CHANGED_COUNT} paths, areas: ${TOP_AREAS})")
fi
if has_changed '^LongMemEval/src/evaluation/evaluate_qa\.py$'; then
  auto_conv_parts+=("Evaluation script was updated for local env compatibility")
fi
if has_changed '^anna_longmemeval_bridge/README\.md$|^ldagent_longmemeval_bridge/README\.md$|^memos_longmemeval_bridge/README\.md$|^share_longmemeval_bridge/README\.md$'; then
  auto_conv_parts+=("Bridge evaluation commands were made machine-agnostic")
fi
if has_changed '^share_longmemeval_bridge/run_infer\.py$'; then
  auto_conv_parts+=("SHARE bridge memory settings were adjusted")
fi
if has_changed '^scripts/handoff\.sh$|^ops/HANDOFF\.md$|^ops/CHANGELOG\.md$'; then
  auto_conv_parts+=("Cross-machine handoff workflow was introduced")
fi
if [[ ${#auto_conv_parts[@]} -eq 0 ]]; then
  auto_conv_parts+=("No clear conversational milestones inferred from local diff")
fi
AUTO_CONV_SUMMARY="$(join_with_semicolon "${auto_conv_parts[@]}")"
AUTO_CONV_SUMMARY="$(trim_one_line "${AUTO_CONV_SUMMARY}")"

declare -a auto_progress_parts
auto_progress_parts+=("$(progress_from_preds "Anna" "${REPO_ROOT}/LongMemEval/preds_anna_s_50.jsonl" 50)")
if [[ -f "${REPO_ROOT}/LongMemEval/preds_share_s_50_nocap.jsonl" ]]; then
  auto_progress_parts+=("$(progress_from_preds "SHARE" "${REPO_ROOT}/LongMemEval/preds_share_s_50_nocap.jsonl" 50)")
elif [[ -f "${REPO_ROOT}/LongMemEval/preds_share_s_50.jsonl" ]]; then
  auto_progress_parts+=("$(progress_from_preds "SHARE" "${REPO_ROOT}/LongMemEval/preds_share_s_50.jsonl" 50)")
elif [[ -f "${REPO_ROOT}/LongMemEval/preds_share_s_50_max10MemoryCap.jsonl" ]]; then
  auto_progress_parts+=("$(progress_from_preds "SHARE" "${REPO_ROOT}/LongMemEval/preds_share_s_50_max10MemoryCap.jsonl" 50)")
else
  auto_progress_parts+=("SHARE not started")
fi
auto_progress_parts+=("$(progress_from_preds "MemoryOS" "${REPO_ROOT}/LongMemEval/preds_memoryos_s_50.jsonl" 50)")
auto_progress_parts+=("$(progress_from_preds "LD-Agent" "${REPO_ROOT}/LongMemEval/preds_ldagent_s_50.jsonl" 50)")
if [[ -f "${REPO_ROOT}/LongMemEval/memoryos_progress_record.txt" ]]; then
  MEMOS_PROGRESS="$(grep -E '^progress:' "${REPO_ROOT}/LongMemEval/memoryos_progress_record.txt" | head -n1 | cut -d' ' -f2- || true)"
  if [[ -n "${MEMOS_PROGRESS}" ]]; then
    auto_progress_parts+=("MemoryOS runtime checkpoint ${MEMOS_PROGRESS} (memoryos_progress_record.txt)")
  fi
fi
AUTO_PROGRESS_TEXT="$(join_with_semicolon "${auto_progress_parts[@]}")"
AUTO_PROGRESS_TEXT="$(trim_one_line "${AUTO_PROGRESS_TEXT}")"

declare -a auto_resolved_parts
if has_changed '^LongMemEval/src/evaluation/evaluate_qa\.py$'; then
  if grep -q 'Loaded env from' "${REPO_ROOT}/LongMemEval/src/evaluation/evaluate_qa.py" && grep -q 'OPENAI_API_KEY' "${REPO_ROOT}/LongMemEval/src/evaluation/evaluate_qa.py"; then
    auto_resolved_parts+=("evaluate_qa now auto-loads .env and checks OPENAI_API_KEY")
  fi
  if grep -q "unexpected keyword argument 'proxies'" "${REPO_ROOT}/LongMemEval/src/evaluation/evaluate_qa.py" && grep -q 'httpx.Client' "${REPO_ROOT}/LongMemEval/src/evaluation/evaluate_qa.py"; then
    auto_resolved_parts+=("evaluate_qa now falls back to explicit httpx client for proxies compatibility")
  fi
fi
if has_changed '^anna_longmemeval_bridge/README\.md$|^ldagent_longmemeval_bridge/README\.md$|^memos_longmemeval_bridge/README\.md$|^share_longmemeval_bridge/README\.md$'; then
  auto_resolved_parts+=("Bridge evaluation commands now use REPO_ROOT instead of machine-specific absolute paths")
fi
if has_changed '^share_longmemeval_bridge/run_infer\.py$'; then
  if grep -nA4 -- '--memory-max-items' "${REPO_ROOT}/share_longmemeval_bridge/run_infer.py" | grep -q 'default=0'; then
    auto_resolved_parts+=("SHARE memory cap default now disables per-bucket item limit (0 means no cap)")
  fi
fi
if [[ ${#auto_resolved_parts[@]} -eq 0 ]]; then
  auto_resolved_parts+=("No explicit resolved issue inferred from current local changes")
fi
AUTO_RESOLVED_TEXT="$(join_with_semicolon "${auto_resolved_parts[@]}")"
AUTO_RESOLVED_TEXT="$(trim_one_line "${AUTO_RESOLVED_TEXT}")"

declare -a auto_pending_parts
if [[ "${CHANGED_COUNT}" -gt 0 ]]; then
  auto_pending_parts+=("Review and commit/push local changes so other machines can pull the same state")
fi
if [[ -f "${REPO_ROOT}/LongMemEval/memoryos_progress_record.txt" ]]; then
  MEMOS_PROGRESS="$(grep -E '^progress:' "${REPO_ROOT}/LongMemEval/memoryos_progress_record.txt" | head -n1 | cut -d' ' -f2- || true)"
  if [[ -n "${MEMOS_PROGRESS}" ]] && [[ "${MEMOS_PROGRESS}" != "50/50" ]]; then
    auto_pending_parts+=("MemoryOS run appears incomplete (${MEMOS_PROGRESS}); resume or finish before final comparison")
  fi
fi
if [[ ! -f "${REPO_ROOT}/LongMemEval/preds_ldagent_s_50.jsonl" ]]; then
  auto_pending_parts+=("LD-Agent 50-question run not found; run it for 4-agent comparison")
fi
declare -a eval_missing
for preds in \
  "${REPO_ROOT}/LongMemEval/preds_anna_s_50.jsonl" \
  "${REPO_ROOT}/LongMemEval/preds_share_s_50.jsonl" \
  "${REPO_ROOT}/LongMemEval/preds_share_s_50_nocap.jsonl" \
  "${REPO_ROOT}/LongMemEval/preds_memoryos_s_50.jsonl" \
  "${REPO_ROOT}/LongMemEval/preds_ldagent_s_50.jsonl"; do
  if [[ -f "${preds}" ]] && [[ ! -f "${preds}.eval-results-gpt-4o" ]]; then
    eval_missing+=("$(basename "${preds}")")
  fi
done
if [[ ${#eval_missing[@]} -gt 0 ]]; then
  auto_pending_parts+=("Evaluation logs missing for: $(join_with_semicolon "${eval_missing[@]}")")
fi
if [[ ${#auto_pending_parts[@]} -eq 0 ]]; then
  auto_pending_parts+=("No obvious pending decision inferred from local repo state")
fi
AUTO_PENDING_TEXT="$(join_with_semicolon "${auto_pending_parts[@]}")"
AUTO_PENDING_TEXT="$(trim_one_line "${AUTO_PENDING_TEXT}")"

if [[ -z "${NOTE}" ]]; then
  NOTE="${AUTO_NOTE}"
fi
if [[ -z "${CONV_SUMMARY}" ]]; then
  CONV_SUMMARY="${AUTO_CONV_SUMMARY}"
fi
if [[ -z "${PROGRESS_TEXT}" ]]; then
  PROGRESS_TEXT="${AUTO_PROGRESS_TEXT}"
fi
if [[ -z "${RESOLVED_TEXT}" ]]; then
  RESOLVED_TEXT="${AUTO_RESOLVED_TEXT}"
fi
if [[ -z "${PENDING_TEXT}" ]]; then
  PENDING_TEXT="${AUTO_PENDING_TEXT}"
fi

ENTRY_HEADER="## Handoff - ${UTC_NOW} - ${MACHINE}"

ENTRY_BODY="$(cat <<EOF
${ENTRY_HEADER}
- Date/Time (UTC): ${UTC_NOW}
- Machine: ${MACHINE}
- Branch + Commit: \`${BRANCH} @ ${COMMIT}\`
- Workspace: \`${REPO_ROOT}\`
- Quick summary: ${NOTE}
- Conversation summary: ${CONV_SUMMARY}
- Current progress: ${PROGRESS_TEXT}
- Resolved issues: ${RESOLVED_TEXT}
- Pending decisions: ${PENDING_TEXT}
- What changed:
- Run status:
- Blockers:
- Next 3 steps:
1. <step 1>
2. <step 2>
3. <step 3>
- Git status snapshot:
\`\`\`
${STATUS_LINES}
\`\`\`
EOF
)"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%s\n' "${ENTRY_BODY}"
  exit 0
fi

mkdir -p "$(dirname "${OUT_FILE}")"
if [[ ! -f "${OUT_FILE}" ]]; then
  cat > "${OUT_FILE}" <<'EOF'
# Cross-Machine Handoff Log

This file is the shared working memory across machines.

EOF
fi

{
  printf '\n'
  printf '%s\n' "${ENTRY_BODY}"
} >> "${OUT_FILE}"

echo "Handoff entry appended to: ${OUT_FILE}"
