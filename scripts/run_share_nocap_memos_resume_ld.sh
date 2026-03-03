#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${WORKDIR}"

DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json}"
SHARE_BRIDGE="${SHARE_BRIDGE:-${WORKDIR}/share_longmemeval_bridge/run_infer.py}"
MEMOS_BRIDGE="${MEMOS_BRIDGE:-${WORKDIR}/memos_longmemeval_bridge/run_infer.py}"
LD_BRIDGE="${LD_BRIDGE:-${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py}"
SHARE_REPO="${SHARE_REPO:-${WORKDIR}/SHARE}"
MEMOS_REPO="${MEMOS_REPO:-${WORKDIR}/MemoryOS}"
LD_REPO="${LD_REPO:-${WORKDIR}/LD-Agent}"

LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
FAIL_FAST="${FAIL_FAST:-1}"
LIMIT="${LIMIT:-0}"

# New SHARE output tag (no-cap run)
SHARE_RUN_TAG="${SHARE_RUN_TAG:-s_50_nocap}"
# Base tag used by current/previous MemoryOS and LD runs
BASE_RUN_TAG="${BASE_RUN_TAG:-s_50}"

SHARE_STRICT_SELECTION_MODE="${SHARE_STRICT_SELECTION_MODE:-qa}"
SHARE_MEMORY_MAX_ITEMS="${SHARE_MEMORY_MAX_ITEMS:-0}"
# If 1, allow resuming MemoryOS even when another MemoryOS process is already running.
FORCE_MEMOS_RESUME_WHILE_RUNNING="${FORCE_MEMOS_RESUME_WHILE_RUNNING:-0}"

SHARE_OUT="${WORKDIR}/LongMemEval/preds_share_${SHARE_RUN_TAG}.jsonl"
SHARE_TRACE="${WORKDIR}/LongMemEval/preds_share_${SHARE_RUN_TAG}.trace.jsonl"

MEMOS_BASE_OUT="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.jsonl"
MEMOS_BASE_TRACE="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.trace.jsonl"
MEMOS_RESUME_OUT="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.resume.jsonl"
MEMOS_RESUME_TRACE="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.resume.trace.jsonl"
MEMOS_MERGED_OUT="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.resumed_merged.jsonl"
MEMOS_MERGED_TRACE="${WORKDIR}/LongMemEval/preds_memoryos_${BASE_RUN_TAG}.resumed_merged.trace.jsonl"

LD_OUT="${WORKDIR}/LongMemEval/preds_ldagent_${BASE_RUN_TAG}.jsonl"
LD_TRACE="${WORKDIR}/LongMemEval/preds_ldagent_${BASE_RUN_TAG}.trace.jsonl"

mkdir -p "${WORKDIR}/LongMemEval"

CONDA_SH=""
for candidate in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done

if [[ -z "${CONDA_SH}" ]]; then
  echo "[ERROR] conda.sh not found." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-False}"

LIMIT_ARGS=()
COMMON_ARGS=(
  --openai-base-url "${OPENAI_BASE_URL}"
  --llm-model "${LLM_MODEL}"
)
if [[ "${FAIL_FAST}" == "1" ]]; then
  COMMON_ARGS+=(--fail-fast)
fi
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

dataset_total() {
  python3 - <<'PY'
import json
from pathlib import Path
p = Path("LongMemEval/data/longmemeval_s_cleaned_50.json")
print(len(json.loads(p.read_text(encoding="utf-8"))))
PY
}

jsonl_count() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    wc -l < "${f}"
  else
    echo 0
  fi
}

last_qid() {
  local f="$1"
  if [[ ! -s "${f}" ]]; then
    echo "N/A"
    return
  fi
  python3 - "$f" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    lines = [ln.strip() for ln in fh if ln.strip()]
obj = json.loads(lines[-1])
print(obj.get("question_id", "N/A"))
PY
}

run_share_nocap() {
  echo "[$(date)] Run SHARE (no memory cap) -> ${SHARE_OUT}"
  conda activate share-lme
  python "${SHARE_BRIDGE}" \
    --share-dir "${SHARE_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${SHARE_OUT}" \
    --trace-jsonl "${SHARE_TRACE}" \
    --strict-selection-mode "${SHARE_STRICT_SELECTION_MODE}" \
    --memory-max-items "${SHARE_MEMORY_MAX_ITEMS}" \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"
}

resume_memoryos_if_needed() {
  local total done last pids
  total="$(dataset_total)"
  done="$(jsonl_count "${MEMOS_BASE_OUT}")"
  last="$(last_qid "${MEMOS_BASE_OUT}")"
  echo "[$(date)] MemoryOS status: ${done}/${total}, last_qid=${last}"

  pids="$(pgrep -f "memos_longmemeval_bridge/run_infer.py" || true)"
  if [[ -n "${pids}" && "${FORCE_MEMOS_RESUME_WHILE_RUNNING}" != "1" ]]; then
    echo "[WARN] Detected running MemoryOS process (PID: ${pids//$'\n'/, })."
    echo "[WARN] Skip MemoryOS resume to avoid concurrent writes. Re-run this script after current MemoryOS finishes."
    return
  fi

  if (( done >= total )); then
    echo "[$(date)] MemoryOS already complete. Skip resume."
    return
  fi

  echo "[$(date)] Resume MemoryOS from offset=${done} -> ${MEMOS_RESUME_OUT}"
  conda activate memos-lme
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python "${MEMOS_BRIDGE}" \
    --memoryos-dir "${MEMOS_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${MEMOS_RESUME_OUT}" \
    --trace-jsonl "${MEMOS_RESUME_TRACE}" \
    --reset-mode reinit \
    --offset "${done}" \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"

  if [[ -s "${MEMOS_BASE_OUT}" ]]; then
    cat "${MEMOS_BASE_OUT}" "${MEMOS_RESUME_OUT}" > "${MEMOS_MERGED_OUT}"
  else
    cp "${MEMOS_RESUME_OUT}" "${MEMOS_MERGED_OUT}"
  fi
  if [[ -s "${MEMOS_BASE_TRACE}" ]]; then
    cat "${MEMOS_BASE_TRACE}" "${MEMOS_RESUME_TRACE}" > "${MEMOS_MERGED_TRACE}"
  else
    cp "${MEMOS_RESUME_TRACE}" "${MEMOS_MERGED_TRACE}"
  fi

  echo "[$(date)] MemoryOS merged outputs:"
  echo "  - ${MEMOS_MERGED_OUT}"
  echo "  - ${MEMOS_MERGED_TRACE}"
}

run_ld_agent() {
  echo "[$(date)] Run LD-Agent -> ${LD_OUT}"
  conda activate ld-lme
  python "${LD_BRIDGE}" \
    --ld-agent-dir "${LD_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${LD_OUT}" \
    --trace-jsonl "${LD_TRACE}" \
    --session-gap-seconds 600 \
    --dist-thres 0.5527 \
    --no-force-flush-before-answer \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"
}

run_share_nocap
resume_memoryos_if_needed
run_ld_agent

echo "[$(date)] Done."
