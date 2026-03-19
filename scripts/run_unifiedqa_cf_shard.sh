#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 14 ]]; then
  echo "Usage: $0 <workdir> <agent> <part_tag> <offset> <limit> <key_var> <run_tag> <llm_model> <openai_base_url> <start_delay_s> <cf_target_scope> <cf_max_writes> <cf_dominance_threshold> <enable_cf_wrapper>" >&2
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
CF_TARGET_SCOPE="${11}"
CF_MAX_WRITES="${12}"
CF_DOMINANCE_THRESHOLD="${13}"
ENABLE_CF_WRAPPER="${14}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
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
OUTPUT_SUFFIX="${RUN_TAG}_${PART_TAG}"

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
  )
fi

echo "============================================================"
echo "[$(date)] agent=${AGENT} part=${PART_TAG} offset=${OFFSET} limit=${LIMIT}"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "key_var=${KEY_VAR}"
echo "enable_cf_wrapper=${ENABLE_CF_WRAPPER}"
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
    python "${WORKDIR}/anna_longmemeval_bridge/run_infer.py" \
      --anna-agent-dir "${WORKDIR}/AnnaAgent" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --disable-full-tertiary-init \
      --disable-need-check \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}" \
      "${CF_ARGS[@]}"
    ;;

  share)
    conda activate share-lme
    python "${WORKDIR}/share_longmemeval_bridge/run_infer.py" \
      --share-dir "${WORKDIR}/SHARE" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --strict-selection-mode qa \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}" \
      "${CF_ARGS[@]}"
    ;;

  memoryos)
    conda activate memos-lme
    python "${WORKDIR}/memos_longmemeval_bridge/run_infer.py" \
      --memoryos-dir "${WORKDIR}/MemoryOS" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --reset-mode reinit \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}" \
      "${CF_ARGS[@]}"
    ;;

  ldagent)
    conda activate ld-lme
    python "${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py" \
      --ld-agent-dir "${WORKDIR}/LD-Agent" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --session-gap-seconds 600 \
      --dist-thres 0.5527 \
      --no-force-flush-before-answer \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}" \
      "${CF_ARGS[@]}"
    ;;

  theanine)
    conda activate theanine-lme
    THEANINE_REPO="${WORKDIR}/Theanine_${PART_TAG}_repo"
    if [[ ! -d "${THEANINE_REPO}" ]]; then
      THEANINE_REPO="${WORKDIR}/Theanine"
    fi
    python "${WORKDIR}/theanine_longmemeval_bridge/run_infer.py" \
      --theanine-dir "${THEANINE_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.trace.jsonl" \
      --llm-model "${LLM_MODEL}" \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}" \
      "${CF_ARGS[@]}"
    ;;

  *)
    echo "Unknown agent '${AGENT}'" >&2
    exit 2
    ;;
esac

echo "[$(date)] Completed agent=${AGENT} part=${PART_TAG}"
