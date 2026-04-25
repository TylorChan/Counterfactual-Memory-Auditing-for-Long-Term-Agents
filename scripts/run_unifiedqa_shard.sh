#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "Usage: $0 <workdir> <agent> <part_tag> <offset> <limit> <key_var> <run_tag> <llm_model> <openai_base_url> <start_delay_s>" >&2
  exit 2
fi

WORKDIR="$1"
AGENT="$2"
PART_TAG="$3"
OFFSET="$4"
LIMIT="$5"
KEY_VAR="$6"
RUN_TAG="$7"
LLM_MODEL="$8"
OPENAI_BASE_URL="$9"
START_DELAY_S="${10}"

source "${WORKDIR}/scripts/conda_bootstrap.sh"
source_conda_sh
export ANONYMIZED_TELEMETRY=False
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

load_env_file() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line#export }"
    if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      local key="${BASH_REMATCH[1]}"
      local value="${BASH_REMATCH[2]}"
      if [[ "${value}" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value}" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      fi
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

load_env_file "${WORKDIR}/.env"

KEY_VALUE="${!KEY_VAR:-${OPENAI_API_KEY:-}}"
if [[ -z "${KEY_VALUE}" ]]; then
  echo "Missing OpenAI key: checked ${KEY_VAR}, then OPENAI_API_KEY." >&2
  exit 1
fi
export OPENAI_API_KEY="${KEY_VALUE}"
export LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED:-1}"
export LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX:-lme-longmemeval}"
export LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG:-1}"

if [[ "${START_DELAY_S}" != "0" ]]; then
  sleep "${START_DELAY_S}"
fi

DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json}"
OUTPUT_SUFFIX="${RUN_TAG}_${PART_TAG}"

COMMON_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${OFFSET}" != "0" ]]; then
  COMMON_ARGS+=(--offset "${OFFSET}")
fi

echo "============================================================"
echo "[$(date)] agent=${AGENT} part=${PART_TAG} offset=${OFFSET} limit=${LIMIT}"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "key_var=${KEY_VAR}"
echo "data_file=${DATA_FILE}"
echo "prompt_cache_enabled=${LME_PROMPT_CACHE_ENABLED}"
echo "prompt_cache_key_prefix=${LME_PROMPT_CACHE_KEY_PREFIX}"
echo "prompt_cache_log=${LME_PROMPT_CACHE_LOG}"
echo "prompt_cache_retention=${LME_PROMPT_CACHE_RETENTION:-}"
if [[ "${AGENT}" == "theanine" ]]; then
  THEANINE_ROOT="${THEANINE_REPO_ROOT:-${WORKDIR}}"
  echo "theanine_repo_candidate=${THEANINE_ROOT}/Theanine_${PART_TAG}_repo"
fi
echo "============================================================"

case "${AGENT}" in
  anna)
    conda activate anna-lme
    cmd=(
      python "${WORKDIR}/anna_longmemeval_bridge/run_infer.py"
      --anna-agent-dir "${WORKDIR}/AnnaAgent"
      --longmemeval-file "${DATA_FILE}"
      --out-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.jsonl"
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.trace.jsonl"
      --openai-base-url "${OPENAI_BASE_URL}"
      --llm-model "${LLM_MODEL}"
      --disable-full-tertiary-init
      --disable-need-check
    )
    if (( ${#COMMON_ARGS[@]} > 0 )); then
      cmd+=("${COMMON_ARGS[@]}")
    fi
    "${cmd[@]}"
    ;;

  share)
    conda activate share-lme
    cmd=(
      python "${WORKDIR}/share_longmemeval_bridge/run_infer.py"
      --share-dir "${WORKDIR}/SHARE"
      --longmemeval-file "${DATA_FILE}"
      --out-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.jsonl"
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.trace.jsonl"
      --openai-base-url "${OPENAI_BASE_URL}"
      --llm-model "${LLM_MODEL}"
      --strict-selection-mode qa
    )
    if (( ${#COMMON_ARGS[@]} > 0 )); then
      cmd+=("${COMMON_ARGS[@]}")
    fi
    "${cmd[@]}"
    ;;

  memoryos)
    conda activate memos-lme
    cmd=(
      python "${WORKDIR}/memos_longmemeval_bridge/run_infer.py"
      --memoryos-dir "${WORKDIR}/MemoryOS"
      --longmemeval-file "${DATA_FILE}"
      --out-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.jsonl"
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.trace.jsonl"
      --openai-base-url "${OPENAI_BASE_URL}"
      --llm-model "${LLM_MODEL}"
      --reset-mode reinit
    )
    if (( ${#COMMON_ARGS[@]} > 0 )); then
      cmd+=("${COMMON_ARGS[@]}")
    fi
    "${cmd[@]}"
    ;;

  ldagent)
    conda activate ld-lme
    cmd=(
      python "${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py"
      --ld-agent-dir "${WORKDIR}/LD-Agent"
      --longmemeval-file "${DATA_FILE}"
      --out-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.jsonl"
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.trace.jsonl"
      --openai-base-url "${OPENAI_BASE_URL}"
      --llm-model "${LLM_MODEL}"
      --session-gap-seconds 600
      --dist-thres 0.5527
      --no-force-flush-before-answer
    )
    if (( ${#COMMON_ARGS[@]} > 0 )); then
      cmd+=("${COMMON_ARGS[@]}")
    fi
    "${cmd[@]}"
    ;;

  theanine)
    conda activate theanine-lme
    THEANINE_ROOT="${THEANINE_REPO_ROOT:-${WORKDIR}}"
    THEANINE_REPO="${THEANINE_ROOT}/Theanine_${PART_TAG}_repo"
    if [[ ! -d "${THEANINE_REPO}" ]]; then
      THEANINE_REPO="${WORKDIR}/Theanine"
    fi
    cmd=(
      python "${WORKDIR}/theanine_longmemeval_bridge/run_infer.py"
      --theanine-dir "${THEANINE_REPO}"
      --longmemeval-file "${DATA_FILE}"
      --out-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.jsonl"
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.trace.jsonl"
      --llm-model "${LLM_MODEL}"
    )
    if (( ${#COMMON_ARGS[@]} > 0 )); then
      cmd+=("${COMMON_ARGS[@]}")
    fi
    "${cmd[@]}"
    ;;

  *)
    echo "Unknown agent '${AGENT}'" >&2
    exit 2
    ;;
esac

echo "[$(date)] Completed agent=${AGENT} part=${PART_TAG}"
