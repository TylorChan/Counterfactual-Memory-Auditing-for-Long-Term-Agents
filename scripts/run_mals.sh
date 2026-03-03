#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-${SCRIPT_DIR}}"
cd "${WORKDIR}"

DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json}"
ANNA_BRIDGE="${ANNA_BRIDGE:-${WORKDIR}/anna_longmemeval_bridge/run_infer.py}"
LD_BRIDGE="${LD_BRIDGE:-${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py}"
SHARE_BRIDGE="${SHARE_BRIDGE:-${WORKDIR}/share_longmemeval_bridge/run_infer.py}"
MEMOS_BRIDGE="${MEMOS_BRIDGE:-${WORKDIR}/memos_longmemeval_bridge/run_infer.py}"
ANNA_REPO="${ANNA_REPO:-${WORKDIR}/AnnaAgent}"
LD_REPO="${LD_REPO:-${WORKDIR}/LD-Agent}"
SHARE_REPO="${SHARE_REPO:-${WORKDIR}/SHARE}"
MEMOS_REPO="${MEMOS_REPO:-${WORKDIR}/MemoryOS}"

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
RUN_TAG="${RUN_TAG:-s_50}"
SHARE_STRICT_SELECTION_MODE="${SHARE_STRICT_SELECTION_MODE:-qa}"
SHARE_MEMORY_MAX_ITEMS="${SHARE_MEMORY_MAX_ITEMS:-0}"
AGENT="${AGENT:-all}"
FAIL_FAST="${FAIL_FAST:-1}"

ANNA_OUT="${WORKDIR}/LongMemEval/preds_anna_${RUN_TAG}.jsonl"
ANNA_TRACE="${WORKDIR}/LongMemEval/preds_anna_${RUN_TAG}.trace.jsonl"
LD_OUT="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.jsonl"
LD_TRACE="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.trace.jsonl"
SHARE_OUT="${WORKDIR}/LongMemEval/preds_share_${RUN_TAG}.jsonl"
SHARE_TRACE="${WORKDIR}/LongMemEval/preds_share_${RUN_TAG}.trace.jsonl"
MEMOS_OUT="${WORKDIR}/LongMemEval/preds_memoryos_${RUN_TAG}.jsonl"
MEMOS_TRACE="${WORKDIR}/LongMemEval/preds_memoryos_${RUN_TAG}.trace.jsonl"

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
  echo "[ERROR] conda.sh not found. Set up conda or export CONDA_SH manually." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-False}"

LIMIT_ARGS=()
OFFSET_ARGS=()
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
if [[ "${OFFSET}" != "0" ]]; then
  OFFSET_ARGS=(--offset "${OFFSET}")
fi

run_anna() {
  echo "[$(date)] AGENT=anna"
  conda activate anna-lme
  python "${ANNA_BRIDGE}" \
    --anna-agent-dir "${ANNA_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${ANNA_OUT}" \
    --trace-jsonl "${ANNA_TRACE}" \
    --disable-full-tertiary-init \
    --disable-need-check \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "${OFFSET_ARGS[@]}"
}

run_share() {
  echo "[$(date)] AGENT=share"
  conda activate share-lme
  python "${SHARE_BRIDGE}" \
    --share-dir "${SHARE_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${SHARE_OUT}" \
    --trace-jsonl "${SHARE_TRACE}" \
    --strict-selection-mode "${SHARE_STRICT_SELECTION_MODE}" \
    --memory-max-items "${SHARE_MEMORY_MAX_ITEMS}" \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "${OFFSET_ARGS[@]}"
}

run_memos() {
  echo "[$(date)] AGENT=memos"
  conda activate memos-lme
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python "${MEMOS_BRIDGE}" \
    --memoryos-dir "${MEMOS_REPO}" \
    --longmemeval-file "${DATA_FILE}" \
    --out-jsonl "${MEMOS_OUT}" \
    --trace-jsonl "${MEMOS_TRACE}" \
    --reset-mode reinit \
    "${COMMON_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "${OFFSET_ARGS[@]}"
}

run_ld() {
  echo "[$(date)] AGENT=ld"
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
    "${LIMIT_ARGS[@]}" \
    "${OFFSET_ARGS[@]}"
}

case "${AGENT}" in
  anna) run_anna ;;
  share) run_share ;;
  memos|memoryos) run_memos ;;
  ld|ld-agent) run_ld ;;
  all)
    run_anna
    run_share
    run_memos
    run_ld
    ;;
  *)
    echo "[ERROR] Unknown AGENT=${AGENT}. Use anna|share|memos|ld|all" >&2
    exit 2
    ;;
esac

echo "[$(date)] Finished AGENT=${AGENT} RUN_TAG=${RUN_TAG}"
