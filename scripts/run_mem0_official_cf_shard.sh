#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: $0 <workdir> <part_tag> <offset> <limit> <key_var> <run_tag> <answerer_model> <start_delay_s> <enable_cf_wrapper>" >&2
  exit 2
fi

WORKDIR="$1"
PART_TAG="$2"
OFFSET="$3"
LIMIT="$4"
KEY_VAR="$5"
RUN_TAG="$6"
ANSWERER_MODEL="$7"
START_DELAY_S="$8"
ENABLE_CF_WRAPPER="$9"

source "${WORKDIR}/scripts/conda_bootstrap.sh"
source_conda_sh

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

KEY_VALUE="${!KEY_VAR:-}"
if [[ -z "${KEY_VALUE}" ]]; then
  echo "Missing OpenAI key variable ${KEY_VAR} in ${WORKDIR}/.env" >&2
  exit 1
fi
if [[ -z "${MEM0_API_KEY:-}" && "${MEM0_OFFICIAL_BACKEND:-cloud}" == "cloud" ]]; then
  echo "Missing MEM0_API_KEY for official Mem0 cloud backend." >&2
  exit 1
fi

export OPENAI_API_KEY="${KEY_VALUE}"
export MEM0_TELEMETRY=false
export ANONYMIZED_TELEMETRY=False
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED:-1}"
export LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX:-lme-longmemeval}"
export LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG:-1}"

if [[ "${START_DELAY_S}" != "0" ]]; then
  sleep "${START_DELAY_S}"
fi

conda activate mem0-lme

DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/LongMemEval}"
OUTPUT_SUFFIX="${RUN_TAG}_${PART_TAG}"
BENCH_DIR="${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}"
BENCH_REF="${MEM0_BENCHMARKS_REF:-f75666d33ef560f0f196746e0e16c515d17e6856}"
BACKEND="${MEM0_OFFICIAL_BACKEND:-cloud}"
TOP_K="${MEM0_OFFICIAL_TOP_K:-200}"
ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF:-200}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
CF_RULE_MODE="${CF_RULE_MODE:-rollback-only}"
RPM="${RPM:-200}"
PROJECT_NAME="${PROJECT_NAME:-mem0-official-cf}"
MAX_ADD_RETRIES="${MEM0_OFFICIAL_MAX_ADD_RETRIES:-8}"
ADD_EVENT_RETRIES="${MEM0_OFFICIAL_ADD_EVENT_RETRIES:-3}"
RETRY_DELAY="${MEM0_OFFICIAL_RETRY_DELAY:-10}"
REQUEST_TIMEOUT="${MEM0_OFFICIAL_REQUEST_TIMEOUT:-300}"
EVENT_POLL_TIMEOUT="${MEM0_OFFICIAL_EVENT_POLL_TIMEOUT:-1800}"

mkdir -p "${OUTPUT_ROOT}"

LIMIT_ARGS=()
OFFSET_ARGS=()
CF_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi
if [[ "${OFFSET}" != "0" ]]; then
  OFFSET_ARGS=(--offset "${OFFSET}")
fi
if [[ "${ENABLE_CF_WRAPPER}" == "1" ]]; then
  CF_ARGS=(
    --enable-cf-wrapper
    --cf-target-scope "${CF_TARGET_SCOPE}"
    --cf-max-writes "${CF_MAX_WRITES}"
    --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}"
    --cf-rule-mode "${CF_RULE_MODE}"
  )
else
  CF_ARGS=(--cf-rule-mode "${CF_RULE_MODE}")
fi

HOST_ARGS=()
if [[ -n "${MEM0_HOST:-}" ]]; then
  HOST_ARGS=(--mem0-host "${MEM0_HOST}")
fi
BASE_URL_ARGS=()
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  BASE_URL_ARGS=(--openai-base-url "${OPENAI_BASE_URL}")
fi
CLEANUP_ARGS=()
if [[ "${MEM0_OFFICIAL_CLEANUP_USERS:-0}" == "1" ]]; then
  CLEANUP_ARGS=(--cleanup-users)
fi

cat <<INFO
============================================================
[$(date)] agent=mem0_official part=${PART_TAG} offset=${OFFSET} limit=${LIMIT}
host=$(hostname)
backend=${BACKEND}
bench_dir=${BENCH_DIR}
bench_ref=${BENCH_REF}
answerer_model=${ANSWERER_MODEL}
top_k=${TOP_K}
answer_cutoff=${ANSWER_CUTOFF}
key_var=${KEY_VAR}
data_file=${DATA_FILE}
output_root=${OUTPUT_ROOT}
enable_cf_wrapper=${ENABLE_CF_WRAPPER}
cf_rule_mode=${CF_RULE_MODE}
cf_target_scope=${CF_TARGET_SCOPE}
cf_max_writes=${CF_MAX_WRITES}
max_add_retries=${MAX_ADD_RETRIES}
add_event_retries=${ADD_EVENT_RETRIES}
retry_delay=${RETRY_DELAY}
request_timeout=${REQUEST_TIMEOUT}
event_poll_timeout=${EVENT_POLL_TIMEOUT}
============================================================
INFO

python "${WORKDIR}/mem0_official_longmemeval_bridge/run_infer.py" \
  --memory-benchmarks-dir "${BENCH_DIR}" \
  --memory-benchmarks-ref "${BENCH_REF}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${OUTPUT_SUFFIX}.jsonl" \
  --trace-jsonl "${OUTPUT_ROOT}/preds_mem0_official_${OUTPUT_SUFFIX}.trace.jsonl" \
  --project-name "${PROJECT_NAME}" \
  --backend "${BACKEND}" \
  --answerer-model "${ANSWERER_MODEL}" \
  --top-k "${TOP_K}" \
  --answer-cutoff "${ANSWER_CUTOFF}" \
  --rpm "${RPM}" \
  --max-add-retries "${MAX_ADD_RETRIES}" \
  --add-event-retries "${ADD_EVENT_RETRIES}" \
  --retry-delay "${RETRY_DELAY}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --event-poll-timeout "${EVENT_POLL_TIMEOUT}" \
  --fail-fast \
  "${HOST_ARGS[@]}" \
  "${BASE_URL_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}" \
  "${CLEANUP_ARGS[@]}" \
  "${CF_ARGS[@]}"

echo "[$(date)] Completed mem0_official part=${PART_TAG}"
