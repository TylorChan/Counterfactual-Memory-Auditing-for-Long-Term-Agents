#!/bin/bash -l
# Single-node GNU Parallel launcher for the unified-QA LongMemEval baseline.
# This follows MSI's recommended pattern for many independent commands inside
# one allocation, avoiding job-array staggering and avoiding repeated srun
# step-creation contention.

#SBATCH --job-name=lme_uqap
#SBATCH --partition=msismall
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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
RUN_TAG="${RUN_TAG:-s_50_unifiedqa_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
COMMANDS_FILE="${WORKDIR}/logs/lme_uqap_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${WORKDIR}/logs/lme_uqap_${SLURM_JOB_ID}.parallel.log"

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
echo "[$(date)] job=${SLURM_JOB_ID} launching unified-QA shards with GNU Parallel"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "run_tag=${RUN_TAG}"
echo "llm_model=${LLM_MODEL}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-6}"
echo "parallel_jobs=${JOBS}"
echo "============================================================"

rm -f "${COMMANDS_FILE}" "${PARALLEL_JOBLOG}"

i=0
for spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r AGENT PART_TAG OFFSET LIMIT KEY_VAR <<< "${spec}"
  DELAY=$((i * STAGGER_SECONDS))
  printf "%q " bash "${WORKDIR}/scripts/run_unifiedqa_shard.sh" \
    "${WORKDIR}" \
    "${AGENT}" \
    "${PART_TAG}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${RUN_TAG}" \
    "${LLM_MODEL}" \
    "${OPENAI_BASE_URL}" \
    "${DELAY}" >> "${COMMANDS_FILE}"
  printf "> %q 2> %q\n" \
    "${WORKDIR}/logs/lme_uqap_${SLURM_JOB_ID}_${AGENT}_${PART_TAG}.out" \
    "${WORKDIR}/logs/lme_uqap_${SLURM_JOB_ID}_${AGENT}_${PART_TAG}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
