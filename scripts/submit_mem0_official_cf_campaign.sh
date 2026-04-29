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
SHARDS="${SHARDS:-10}"
KEY_SLOTS="${KEY_SLOTS:-10}"
ANSWERER_MODEL="${ANSWERER_MODEL:-gpt-5}"
MEM0_OFFICIAL_BACKEND="${MEM0_OFFICIAL_BACKEND:-cloud}"
MEM0_OFFICIAL_TOP_K="${MEM0_OFFICIAL_TOP_K:-200}"
MEM0_OFFICIAL_ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF:-200}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
CF_RULE_MODE="${CF_RULE_MODE:-rollback-only}"
LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED:-1}"
LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX:-lme-longmemeval}"
LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG:-1}"

if [[ "${MEM0_OFFICIAL_BACKEND}" == "cloud" && -z "${MEM0_API_KEY:-}" ]]; then
  echo "Missing MEM0_API_KEY for MEM0_OFFICIAL_BACKEND=cloud." >&2
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
RUN_TAG="${RUN_TAG:-s_100_mem0_official_cf_${STAMP}}"
MANIFEST="${CAMPAIGN_DIR}/submission_manifest.txt"
mkdir -p "${CAMPAIGN_DIR}" "${LOG_DIR}"

{
  echo "campaign_dir=${CAMPAIGN_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "agent=mem0_official"
  echo "backend=${MEM0_OFFICIAL_BACKEND}"
  echo "answerer_model=${ANSWERER_MODEL}"
  echo "data_file=${DATA_FILE}"
  echo "total_questions=${TOTAL_QUESTIONS}"
  echo "shards=${SHARDS}"
  echo "key_slots=${KEY_SLOTS}"
  echo "top_k=${MEM0_OFFICIAL_TOP_K}"
  echo "answer_cutoff=${MEM0_OFFICIAL_ANSWER_CUTOFF}"
  echo "enable_cf_wrapper=${ENABLE_CF_WRAPPER}"
  echo "cf_rule_mode=${CF_RULE_MODE}"
  echo "cf_target_scope=${CF_TARGET_SCOPE}"
  echo "cf_max_writes=${CF_MAX_WRITES}"
  echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
  echo "prompt_cache_enabled=${LME_PROMPT_CACHE_ENABLED}"
  echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${MANIFEST}"

output="$(
  RUN_TAG="${RUN_TAG}" OUTPUT_ROOT="${CAMPAIGN_DIR}" LOG_ROOT="${LOG_DIR}" DATA_FILE="${DATA_FILE}" TOTAL_QUESTIONS="${TOTAL_QUESTIONS}" SHARDS="${SHARDS}" KEY_SLOTS="${KEY_SLOTS}" ANSWERER_MODEL="${ANSWERER_MODEL}" MEM0_OFFICIAL_BACKEND="${MEM0_OFFICIAL_BACKEND}" MEM0_OFFICIAL_TOP_K="${MEM0_OFFICIAL_TOP_K}" MEM0_OFFICIAL_ANSWER_CUTOFF="${MEM0_OFFICIAL_ANSWER_CUTOFF}" ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER}" CF_TARGET_SCOPE="${CF_TARGET_SCOPE}" CF_MAX_WRITES="${CF_MAX_WRITES}" CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD}" CF_RULE_MODE="${CF_RULE_MODE}" LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED}" LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX}" LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG}" \
    sbatch \
    --output "${LOG_DIR}/%x_%j.out" \
    --error "${LOG_DIR}/%x_%j.err" \
    "${WORKDIR}/run_mem0_official_longmemeval_cf_10way.slurm"
)"

echo "run_mem0_official_longmemeval_cf_10way.slurm: ${output}" | tee -a "${MANIFEST}"
echo "Official Mem0 CF job submitted."
echo "Campaign outputs will be written under: ${CAMPAIGN_DIR}"
