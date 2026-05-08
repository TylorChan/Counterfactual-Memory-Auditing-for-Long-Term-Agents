#!/usr/bin/env bash
set -euo pipefail

# Official Mem0 94%-style LongMemEval CF50 campaign.
# This script intentionally has all experiment settings in one place so it can
# be launched without command-line arguments.

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

# -----------------------------
# Fixed experiment configuration
# -----------------------------
DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
TOTAL_QUESTIONS="50"
SHARDS="10"
KEY_SLOTS="10"
ANSWERER_MODEL="gpt-5"
MEM0_OFFICIAL_BACKEND="cloud"
MEM0_OFFICIAL_TOP_K="200"
MEM0_OFFICIAL_ANSWER_CUTOFF="200"
MEM0_OFFICIAL_MAX_ADD_RETRIES="8"
MEM0_OFFICIAL_ADD_EVENT_RETRIES="3"
MEM0_OFFICIAL_RETRY_DELAY="10"
MEM0_OFFICIAL_REQUEST_TIMEOUT="300"
MEM0_OFFICIAL_EVENT_POLL_TIMEOUT="1800"
ENABLE_CF_WRAPPER="1"
CF_RULE_MODE="rollback-only"
CF_TARGET_SCOPE="prompt"
CF_MAX_WRITES="3"
CF_DOMINANCE_THRESHOLD="0.75"
LME_PROMPT_CACHE_ENABLED="1"
LME_PROMPT_CACHE_KEY_PREFIX="lme-longmemeval"
LME_PROMPT_CACHE_LOG="1"
STAGGER_SECONDS="30"
JOBS="10"
RUN_TAG_PREFIX="s_50_mem0_official_cf_rb3"

# -----------------------------
# Local .env parser without sourcing arbitrary shell code
# -----------------------------
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

# -----------------------------
# Preflight checks
# -----------------------------
if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Missing data file: ${DATA_FILE}" >&2
  exit 1
fi
if [[ -z "${MEM0_API_KEY:-}" ]]; then
  echo "Missing MEM0_API_KEY in ${WORKDIR}/.env. Required for Mem0 official cloud/platform setup." >&2
  exit 1
fi
for idx in $(seq 1 "${KEY_SLOTS}"); do
  key_var="OPENAI_API_KEY_${idx}"
  if [[ -z "${!key_var:-}" ]]; then
    echo "Missing ${key_var} in ${WORKDIR}/.env" >&2
    exit 1
  fi
done

if [[ ! -d "${MEM0_BENCHMARKS_DIR:-${WORKDIR}/external/memory-benchmarks}" ]]; then
  bash "${WORKDIR}/scripts/setup_mem0_official_benchmarks.sh"
fi

STAMP="$(date +%m_%d_%H_%M)"
CAMPAIGN_DIR="${WORKDIR}/LongMemEval/${STAMP}"
LOG_DIR="${CAMPAIGN_DIR}/logs"
RUN_TAG="${RUN_TAG_PREFIX}_${STAMP}"
MANIFEST="${CAMPAIGN_DIR}/submission_manifest.txt"
mkdir -p "${CAMPAIGN_DIR}" "${LOG_DIR}"

{
  echo "campaign_dir=${CAMPAIGN_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "agent=mem0_official"
  echo "protocol=official_mem0_memory_benchmarks_true_write_time_rollback"
  echo "slurm_partition=msilong"
  echo "slurm_time=7-00:00:00"
  echo "slurm_ntasks=10"
  echo "slurm_mem=48g"
  echo "backend=${MEM0_OFFICIAL_BACKEND}"
  echo "answerer_model=${ANSWERER_MODEL}"
  echo "data_file=${DATA_FILE}"
  echo "total_questions=${TOTAL_QUESTIONS}"
  echo "shards=${SHARDS}"
  echo "jobs=${JOBS}"
  echo "key_slots=${KEY_SLOTS}"
  echo "top_k=${MEM0_OFFICIAL_TOP_K}"
  echo "answer_cutoff=${MEM0_OFFICIAL_ANSWER_CUTOFF}"
  echo "max_add_retries=${MEM0_OFFICIAL_MAX_ADD_RETRIES}"
  echo "add_event_retries=${MEM0_OFFICIAL_ADD_EVENT_RETRIES}"
  echo "retry_delay=${MEM0_OFFICIAL_RETRY_DELAY}"
  echo "request_timeout=${MEM0_OFFICIAL_REQUEST_TIMEOUT}"
  echo "event_poll_timeout=${MEM0_OFFICIAL_EVENT_POLL_TIMEOUT}"
  echo "enable_cf_wrapper=${ENABLE_CF_WRAPPER}"
  echo "cf_rule_mode=${CF_RULE_MODE}"
  echo "cf_target_scope=${CF_TARGET_SCOPE}"
  echo "cf_max_writes=${CF_MAX_WRITES}"
  echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
  echo "prompt_cache_enabled=${LME_PROMPT_CACHE_ENABLED}"
  echo "prompt_cache_key_prefix=${LME_PROMPT_CACHE_KEY_PREFIX}"
  echo "prompt_cache_log=${LME_PROMPT_CACHE_LOG}"
  echo "estimated_add_requests_warning=50q_with_cf_max_writes_3_is_approximately_50k_adds"
  echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${MANIFEST}"

output="$(
  RUN_TAG="${RUN_TAG}" \
  OUTPUT_ROOT="${CAMPAIGN_DIR}" \
  LOG_ROOT="${LOG_DIR}" \
  DATA_FILE="${DATA_FILE}" \
  TOTAL_QUESTIONS="${TOTAL_QUESTIONS}" \
  SHARDS="${SHARDS}" \
  KEY_SLOTS="${KEY_SLOTS}" \
  ANSWERER_MODEL="${ANSWERER_MODEL}" \
  MEM0_OFFICIAL_BACKEND="${MEM0_OFFICIAL_BACKEND}" \
  MEM0_OFFICIAL_TOP_K="${MEM0_OFFICIAL_TOP_K}" \
  MEM0_OFFICIAL_ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF}" \
  MEM0_OFFICIAL_MAX_ADD_RETRIES="${MEM0_OFFICIAL_MAX_ADD_RETRIES}" \
  MEM0_OFFICIAL_ADD_EVENT_RETRIES="${MEM0_OFFICIAL_ADD_EVENT_RETRIES}" \
  MEM0_OFFICIAL_RETRY_DELAY="${MEM0_OFFICIAL_RETRY_DELAY}" \
  MEM0_OFFICIAL_REQUEST_TIMEOUT="${MEM0_OFFICIAL_REQUEST_TIMEOUT}" \
  MEM0_OFFICIAL_EVENT_POLL_TIMEOUT="${MEM0_OFFICIAL_EVENT_POLL_TIMEOUT}" \
  ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER}" \
  CF_TARGET_SCOPE="${CF_TARGET_SCOPE}" \
  CF_MAX_WRITES="${CF_MAX_WRITES}" \
  CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD}" \
  CF_RULE_MODE="${CF_RULE_MODE}" \
  LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED}" \
  LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX}" \
  LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG}" \
  STAGGER_SECONDS="${STAGGER_SECONDS}" \
  JOBS="${JOBS}" \
    sbatch \
    --output "${LOG_DIR}/%x_%j.out" \
    --error "${LOG_DIR}/%x_%j.err" \
    "${WORKDIR}/slurms/run_mem0_official_longmemeval_cf_10way.slurm"
)"

echo "run_mem0_official_longmemeval_cf_10way.slurm: ${output}" | tee -a "${MANIFEST}"
echo "Official Mem0 CF50 job submitted."
echo "Campaign outputs will be written under: ${CAMPAIGN_DIR}"
echo "Logs will be written under: ${LOG_DIR}"
