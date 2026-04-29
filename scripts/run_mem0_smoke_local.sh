#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"

TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-10}"
OFFSET="${OFFSET:-0}"
PART_TAG="${PART_TAG:-smoke10}"
KEY_VAR="${KEY_VAR:-OPENAI_API_KEY_1}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-candidate}"
CF_MAX_WRITES="${CF_MAX_WRITES:-1}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
MEM0_TOP_K="${MEM0_TOP_K:-50}"
DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
STAMP="$(date +%m_%d_%H_%M)"
RUN_TAG="${RUN_TAG:-mem0_smoke_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/LongMemEval/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

export DATA_FILE OUTPUT_ROOT LOG_ROOT MEM0_TOP_K
export LME_PROMPT_CACHE_ENABLED="${LME_PROMPT_CACHE_ENABLED:-1}"
export LME_PROMPT_CACHE_KEY_PREFIX="${LME_PROMPT_CACHE_KEY_PREFIX:-lme-longmemeval}"
export LME_PROMPT_CACHE_LOG="${LME_PROMPT_CACHE_LOG:-1}"

cat > "${OUTPUT_ROOT}/manifest.txt" <<EOF
run_tag=${RUN_TAG}
output_root=${OUTPUT_ROOT}
log_root=${LOG_ROOT}
data_file=${DATA_FILE}
total_questions=${TOTAL_QUESTIONS}
offset=${OFFSET}
part_tag=${PART_TAG}
key_var=${KEY_VAR}
llm_model=${LLM_MODEL}
openai_base_url=${OPENAI_BASE_URL}
cf_target_scope=${CF_TARGET_SCOPE}
cf_max_writes=${CF_MAX_WRITES}
cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}
enable_cf_wrapper=${ENABLE_CF_WRAPPER}
mem0_top_k=${MEM0_TOP_K}
started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
EOF

echo "Running mem0 smoke test"
echo "output_root=${OUTPUT_ROOT}"
echo "log_root=${LOG_ROOT}"

bash "${WORKDIR}/scripts/run_unifiedqa_cf_shard.sh" \
  "${WORKDIR}" \
  mem0 \
  "${PART_TAG}" \
  "${OFFSET}" \
  "${TOTAL_QUESTIONS}" \
  "${KEY_VAR}" \
  "${RUN_TAG}" \
  "${LLM_MODEL}" \
  "${OPENAI_BASE_URL}" \
  0 \
  "${CF_TARGET_SCOPE}" \
  "${CF_MAX_WRITES}" \
  "${CF_DOMINANCE_THRESHOLD}" \
  "${ENABLE_CF_WRAPPER}" \
  > "${LOG_ROOT}/mem0_smoke_${PART_TAG}.out" \
  2> "${LOG_ROOT}/mem0_smoke_${PART_TAG}.err"

cat >> "${OUTPUT_ROOT}/manifest.txt" <<EOF
finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
preds=${OUTPUT_ROOT}/preds_mem0_${RUN_TAG}_${PART_TAG}.jsonl
trace=${OUTPUT_ROOT}/preds_mem0_${RUN_TAG}_${PART_TAG}.trace.jsonl
cf_runs=${OUTPUT_ROOT}/preds_mem0_${RUN_TAG}_${PART_TAG}.trace.cf_runs.jsonl
cf_queries=${OUTPUT_ROOT}/preds_mem0_${RUN_TAG}_${PART_TAG}.trace.cf_queries.jsonl
EOF

echo "Done. Outputs:"
echo "${OUTPUT_ROOT}"
echo "Logs:"
echo "${LOG_ROOT}/mem0_smoke_${PART_TAG}.out"
echo "${LOG_ROOT}/mem0_smoke_${PART_TAG}.err"
