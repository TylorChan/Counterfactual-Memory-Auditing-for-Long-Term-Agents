#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

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

DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-100}"
SHARDS="${SHARDS:-15}"
KEY_SLOTS="${KEY_SLOTS:-${SHARDS}}"
LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED:-1}"
LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX:-lme-longmemeval}"
LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG:-1}"

for idx in $(seq 1 "${KEY_SLOTS}"); do
  key_var="OPENAI_API_KEY_${idx}"
  if [[ -z "${!key_var:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Missing ${key_var} and OPENAI_API_KEY in ${WORKDIR}/.env" >&2
    exit 1
  elif [[ -z "${!key_var:-}" ]]; then
    echo "Warning: missing ${key_var}; shard(s) using it will fall back to OPENAI_API_KEY." >&2
  fi
done

STAMP="$(date +%m_%d_%H_%M)"
CAMPAIGN_DIR="${WORKDIR}/LongMemEval/${STAMP}"
LOG_DIR="${CAMPAIGN_DIR}/logs"
RUN_TAG="${RUN_TAG:-s_100_unifiedqa_cf_${STAMP}}"
MANIFEST="${CAMPAIGN_DIR}/submission_manifest.txt"

mkdir -p "${CAMPAIGN_DIR}"
mkdir -p "${LOG_DIR}"

{
  echo "campaign_dir=${CAMPAIGN_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "llm_model=${LLM_MODEL:-gpt-4o-mini}"
  echo "data_file=${DATA_FILE}"
  echo "total_questions=${TOTAL_QUESTIONS}"
  echo "shards=${SHARDS}"
  echo "key_slots=${KEY_SLOTS}"
  echo "prompt_cache_enabled=${LME_PROMPT_CACHE_ENABLED}"
  echo "prompt_cache_key_prefix=${LME_PROMPT_CACHE_KEY_PREFIX}"
  echo "prompt_cache_log=${LME_PROMPT_CACHE_LOG}"
  echo "prompt_cache_retention=${LME_PROMPT_CACHE_RETENTION:-}"
  echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${MANIFEST}"

submit_job() {
  local script_name="$1"
  local output
  output="$(
    RUN_TAG="${RUN_TAG}" OUTPUT_ROOT="${CAMPAIGN_DIR}" LOG_ROOT="${LOG_DIR}" DATA_FILE="${DATA_FILE}" TOTAL_QUESTIONS="${TOTAL_QUESTIONS}" SHARDS="${SHARDS}" KEY_SLOTS="${KEY_SLOTS}" LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED}" LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX}" LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG}" LME_PROMPT_CACHE_RETENTION="${LME_PROMPT_CACHE_RETENTION:-}" \
      sbatch \
      --output "${LOG_DIR}/%x_%j.out" \
      --error "${LOG_DIR}/%x_%j.err" \
      "${WORKDIR}/${script_name}"
  )"
  echo "${script_name}: ${output}" | tee -a "${MANIFEST}"
}

submit_job "run_anna_unifiedqa_cf_5x10.slurm"
submit_job "run_memoryos_unifiedqa_cf_5x10.slurm"
submit_job "run_ldagent_unifiedqa_cf_5x10.slurm"
submit_job "run_share_unifiedqa_cf_5x10.slurm"
submit_job "run_theanine_unifiedqa_cf_5x10.slurm"

echo "All jobs submitted."
echo "Campaign outputs will be written under: ${CAMPAIGN_DIR}"
