#!/bin/bash -l
# MemoryOS-only unified-QA baseline+CF run, split into 15 parallel shards by default.

#SBATCH --job-name=memos_uqacf
#SBATCH --partition=msismall
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --mem=80g
#SBATCH --tmp=18g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs}"
cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}"

module load parallel

DATE_SUFFIX="$(date +%m_%d_%H_%M)"
RUN_TAG="${RUN_TAG:-s_100_unifiedqa_cf_memoryos_${DATE_SUFFIX}}"
DATA_FILE="${DATA_FILE:-${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_100_balanced_seed8980.json}"
TOTAL_QUESTIONS="${TOTAL_QUESTIONS:-100}"
SHARDS="${SHARDS:-15}"
KEY_SLOTS="${KEY_SLOTS:-10}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
JOBS="${JOBS:-${SLURM_NTASKS:-${SHARDS}}}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
COMMANDS_FILE="${LOG_ROOT}/memos_uqacf_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${LOG_ROOT}/memos_uqacf_${SLURM_JOB_ID}.parallel.log"
export DATA_FILE

TASK_SPECS=()
shard_base=$((TOTAL_QUESTIONS / SHARDS))
shard_remainder=$((TOTAL_QUESTIONS % SHARDS))
offset=0
for shard_idx in $(seq 1 "${SHARDS}"); do
  limit="${shard_base}"
  if (( shard_idx <= shard_remainder )); then
    limit=$((limit + 1))
  fi
  key_idx=$(( ((shard_idx - 1) % KEY_SLOTS) + 1 ))
  TASK_SPECS+=("s${shard_idx}|${offset}|${limit}|OPENAI_API_KEY_${key_idx}")
  offset=$((offset + limit))
done

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} launching MemoryOS unified-QA baseline+CF ${SHARDS} shards over ${TOTAL_QUESTIONS} questions"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "run_tag=${RUN_TAG}"
echo "llm_model=${LLM_MODEL}"
echo "data_file=${DATA_FILE}"
echo "parallel_jobs=${JOBS}"
echo "key_slots=${KEY_SLOTS}"
echo "enable_cf_wrapper=${ENABLE_CF_WRAPPER}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
echo "============================================================"

rm -f "${COMMANDS_FILE}" "${PARALLEL_JOBLOG}"

i=0
for spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r PART_TAG OFFSET LIMIT KEY_VAR <<< "${spec}"
  DELAY=$((i * STAGGER_SECONDS))
  printf "%q " bash "${WORKDIR}/scripts/run_unifiedqa_cf_shard.sh" \
    "${WORKDIR}" \
    "memoryos" \
    "${PART_TAG}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${RUN_TAG}" \
    "${LLM_MODEL}" \
    "${OPENAI_BASE_URL}" \
    "${DELAY}" \
    "${CF_TARGET_SCOPE}" \
    "${CF_MAX_WRITES}" \
    "${CF_DOMINANCE_THRESHOLD}" \
    "${ENABLE_CF_WRAPPER}" >> "${COMMANDS_FILE}"
  printf "> %q 2> %q\n" \
    "${LOG_ROOT}/memos_uqacf_${SLURM_JOB_ID}_${PART_TAG}.out" \
    "${LOG_ROOT}/memos_uqacf_${SLURM_JOB_ID}_${PART_TAG}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
