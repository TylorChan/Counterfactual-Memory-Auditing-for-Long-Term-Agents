#!/bin/bash -l
# Single-node GNU Parallel launcher for CF-only runs from an existing baseline snapshot.

#SBATCH --job-name=lme_uqacfo
#SBATCH --partition=msismall
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --tmp=12g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"
mkdir -p logs

module load parallel

DATE_SUFFIX="$(date +%m_%d_%H_%M)"
BASELINE_TAG="${BASELINE_TAG:-03_14_22_46}"
CF_TAG="${CF_TAG:-cfonly_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-30}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-8}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
COMMANDS_FILE="${WORKDIR}/logs/lme_uqacfo_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${WORKDIR}/logs/lme_uqacfo_${SLURM_JOB_ID}.parallel.log"

TASK_SPECS=(
  "theanine|p1|0|25|OPENAI_API_KEY_1"
  "theanine|p2|25|25|OPENAI_API_KEY_2"
  "share|p1|0|25|OPENAI_API_KEY_3"
  "share|p2|25|25|OPENAI_API_KEY_4"
  "memoryos|p1|0|25|OPENAI_API_KEY_5"
  "memoryos|p2|25|25|OPENAI_API_KEY"
  "ldagent|p1|0|25|OPENAI_API_KEY_5"
  "ldagent|p2|25|25|OPENAI_API_KEY_4"
  "anna|full|0|0|OPENAI_API_KEY_3"
)

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} launching CF-only shards with GNU Parallel"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "baseline_tag=${BASELINE_TAG}"
echo "cf_tag=${CF_TAG}"
echo "llm_model=${LLM_MODEL}"
echo "parallel_jobs=${JOBS}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
echo "============================================================"

rm -f "${COMMANDS_FILE}" "${PARALLEL_JOBLOG}"

i=0
for spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r AGENT PART_TAG OFFSET LIMIT KEY_VAR <<< "${spec}"
  DELAY=$((i * STAGGER_SECONDS))
  printf "%q " bash "${WORKDIR}/scripts/run_unifiedqa_cf_only_shard.sh" \
    "${WORKDIR}" \
    "${AGENT}" \
    "${PART_TAG}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${BASELINE_TAG}" \
    "${CF_TAG}" \
    "${LLM_MODEL}" \
    "${OPENAI_BASE_URL}" \
    "${DELAY}" \
    "${CF_TARGET_SCOPE}" \
    "${CF_MAX_WRITES}" \
    "${CF_DOMINANCE_THRESHOLD}" >> "${COMMANDS_FILE}"
  printf "> %q 2> %q\n" \
    "${WORKDIR}/logs/lme_uqacfo_${SLURM_JOB_ID}_${AGENT}_${PART_TAG}.out" \
    "${WORKDIR}/logs/lme_uqacfo_${SLURM_JOB_ID}_${AGENT}_${PART_TAG}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
