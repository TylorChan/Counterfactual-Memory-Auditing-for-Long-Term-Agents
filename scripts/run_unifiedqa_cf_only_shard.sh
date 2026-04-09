#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 14 ]]; then
  echo "Usage: $0 <workdir> <agent> <part_tag> <offset> <limit> <key_var> <baseline_tag> <cf_tag> <llm_model> <openai_base_url> <start_delay_s> <cf_target_scope> <cf_max_writes> <cf_dominance_threshold>" >&2
  exit 2
fi

WORKDIR="$1"
AGENT="$2"
PART_TAG="$3"
OFFSET="$4"
LIMIT="$5"
KEY_VAR="$6"
BASELINE_TAG="$7"
CF_TAG="$8"
LLM_MODEL="$9"
OPENAI_BASE_URL="${10}"
START_DELAY_S="${11}"
CF_TARGET_SCOPE="${12}"
CF_MAX_WRITES="${13}"
CF_DOMINANCE_THRESHOLD="${14}"

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
    line="${line#export }"
    if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
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

if [[ "${START_DELAY_S}" != "0" ]]; then
  sleep "${START_DELAY_S}"
fi

DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
RUNTIME_DIR="${WORKDIR}/cf_only_runtime/${CF_TAG}/${AGENT}_${PART_TAG}"
mkdir -p "${RUNTIME_DIR}"

if [[ "${AGENT}" == "anna" ]]; then
  BASELINE_TRACE="${WORKDIR}/LongMemEval/preds_anna_s_50_unifiedqa_${BASELINE_TAG}_full.trace.jsonl"
else
  BASELINE_TRACE="${WORKDIR}/LongMemEval/preds_${AGENT}_s_50_unifiedqa_${BASELINE_TAG}_${PART_TAG}.trace.jsonl"
fi

if [[ ! -f "${BASELINE_TRACE}" ]]; then
  echo "Baseline trace not found: ${BASELINE_TRACE}" >&2
  exit 1
fi

echo "============================================================"
echo "[$(date)] CF-only agent=${AGENT} part=${PART_TAG}"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "baseline_tag=${BASELINE_TAG}"
echo "cf_tag=${CF_TAG}"
echo "key_var=${KEY_VAR}"
echo "baseline_trace=${BASELINE_TRACE}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
if [[ "${AGENT}" == "theanine" ]]; then
  echo "theanine_repo_candidate=${WORKDIR}/Theanine_${PART_TAG}_repo"
fi
echo "============================================================"

case "${AGENT}" in
  anna)
    conda activate anna-lme
    python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
      --agent anna \
      --anna-agent-dir "${WORKDIR}/AnnaAgent" \
      --longmemeval-file "${DATA_FILE}" \
      --baseline-trace-jsonl "${BASELINE_TRACE}" \
      --cf-tag "${CF_TAG}" \
      --llm-model "${LLM_MODEL}" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --cf-target-scope "${CF_TARGET_SCOPE}" \
      --cf-max-writes "${CF_MAX_WRITES}" \
      --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
      --runtime-dir "${RUNTIME_DIR}"
    ;;
  share)
    conda activate share-lme
    python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
      --agent share \
      --share-dir "${WORKDIR}/SHARE" \
      --longmemeval-file "${DATA_FILE}" \
      --baseline-trace-jsonl "${BASELINE_TRACE}" \
      --cf-tag "${CF_TAG}" \
      --llm-model "${LLM_MODEL}" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --cf-target-scope "${CF_TARGET_SCOPE}" \
      --cf-max-writes "${CF_MAX_WRITES}" \
      --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
      --runtime-dir "${RUNTIME_DIR}"
    ;;
  memoryos)
    conda activate memos-lme
    python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
      --agent memoryos \
      --memoryos-dir "${WORKDIR}/MemoryOS" \
      --longmemeval-file "${DATA_FILE}" \
      --baseline-trace-jsonl "${BASELINE_TRACE}" \
      --cf-tag "${CF_TAG}" \
      --llm-model "${LLM_MODEL}" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --cf-target-scope "${CF_TARGET_SCOPE}" \
      --cf-max-writes "${CF_MAX_WRITES}" \
      --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
      --runtime-dir "${RUNTIME_DIR}"
    ;;
  ldagent)
    conda activate ld-lme
    python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
      --agent ldagent \
      --ld-agent-dir "${WORKDIR}/LD-Agent" \
      --longmemeval-file "${DATA_FILE}" \
      --baseline-trace-jsonl "${BASELINE_TRACE}" \
      --cf-tag "${CF_TAG}" \
      --llm-model "${LLM_MODEL}" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --cf-target-scope "${CF_TARGET_SCOPE}" \
      --cf-max-writes "${CF_MAX_WRITES}" \
      --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
      --runtime-dir "${RUNTIME_DIR}"
    ;;
  theanine)
    conda activate theanine-lme
    THEANINE_REPO="${WORKDIR}/Theanine_${PART_TAG}_repo"
    if [[ ! -d "${THEANINE_REPO}" ]]; then
      THEANINE_REPO="${WORKDIR}/Theanine"
    fi
    python "${WORKDIR}/scripts/run_unifiedqa_cf_only.py" \
      --agent theanine \
      --theanine-dir "${THEANINE_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --baseline-trace-jsonl "${BASELINE_TRACE}" \
      --cf-tag "${CF_TAG}" \
      --llm-model "${LLM_MODEL}" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --cf-target-scope "${CF_TARGET_SCOPE}" \
      --cf-max-writes "${CF_MAX_WRITES}" \
      --cf-dominance-threshold "${CF_DOMINANCE_THRESHOLD}" \
      --runtime-dir "${RUNTIME_DIR}"
    ;;
  *)
    echo "Unknown agent '${AGENT}'" >&2
    exit 2
    ;;
esac

echo "[$(date)] Completed CF-only agent=${AGENT} part=${PART_TAG}"
