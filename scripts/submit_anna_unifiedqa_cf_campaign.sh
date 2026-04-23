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

for idx in $(seq 1 10); do
  key_var="OPENAI_API_KEY_${idx}"
  if [[ -z "${!key_var:-}" ]]; then
    echo "Missing ${key_var} in ${WORKDIR}/.env" >&2
    exit 1
  fi
done

STAMP="$(date +%m_%d_%H_%M)"
CAMPAIGN_DIR="${WORKDIR}/LongMemEval/${STAMP}"
LOG_DIR="${CAMPAIGN_DIR}/logs"
RUN_TAG="s_50_unifiedqa_cf_${STAMP}"
MANIFEST="${CAMPAIGN_DIR}/submission_manifest.txt"

mkdir -p "${CAMPAIGN_DIR}"
mkdir -p "${LOG_DIR}"

{
  echo "campaign_dir=${CAMPAIGN_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "agent=anna"
  echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${MANIFEST}"

output="$(
  RUN_TAG="${RUN_TAG}" OUTPUT_ROOT="${CAMPAIGN_DIR}" LOG_ROOT="${LOG_DIR}" \
    sbatch \
    --output "${LOG_DIR}/%x_%j.out" \
    --error "${LOG_DIR}/%x_%j.err" \
    "${WORKDIR}/run_anna_unifiedqa_cf_5x10.slurm"
)"

echo "run_anna_unifiedqa_cf_5x10.slurm: ${output}" | tee -a "${MANIFEST}"
echo "Anna job submitted."
echo "Campaign outputs will be written under: ${CAMPAIGN_DIR}"

