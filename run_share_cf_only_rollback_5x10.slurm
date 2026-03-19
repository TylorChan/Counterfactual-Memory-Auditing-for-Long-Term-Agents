#!/bin/bash -l
# SHARE-only CF rollback run over all 50 questions, split into 5 parallel 10-question shards.

#SBATCH --job-name=share_cf_rb
#SBATCH --partition=msismall
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
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
CF_TAG="${CF_TAG:-share_cf_rb2_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-2}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
COMMANDS_FILE="${WORKDIR}/logs/share_cf_rb_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${WORKDIR}/logs/share_cf_rb_${SLURM_JOB_ID}.parallel.log"

TASK_SPECS=(
  "s1|0|10|OPENAI_API_KEY_1"
  "s2|10|10|OPENAI_API_KEY_2"
  "s3|20|10|OPENAI_API_KEY_3"
  "s4|30|10|OPENAI_API_KEY_4"
  "s5|40|10|OPENAI_API_KEY_5"
)

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} launching SHARE rollback-only CF 5x10"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "baseline_tag=${BASELINE_TAG}"
echo "cf_tag=${CF_TAG}"
echo "parallel_jobs=${JOBS}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
echo "============================================================"

rm -f "${COMMANDS_FILE}" "${PARALLEL_JOBLOG}"

i=0
for spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r SHARD_NAME OFFSET LIMIT KEY_VAR <<< "${spec}"
  DELAY=$((i * STAGGER_SECONDS))
  printf "%q " bash "${WORKDIR}/scripts/run_share_cf_only_5x10_shard.sh" \
    "${WORKDIR}" \
    "${BASELINE_TAG}" \
    "${CF_TAG}" \
    "${SHARD_NAME}" \
    "${OFFSET}" \
    "${LIMIT}" \
    "${KEY_VAR}" \
    "${LLM_MODEL}" \
    "${OPENAI_BASE_URL}" \
    "${DELAY}" \
    "${CF_TARGET_SCOPE}" \
    "${CF_MAX_WRITES}" \
    "${CF_DOMINANCE_THRESHOLD}" >> "${COMMANDS_FILE}"
  printf "> %q 2> %q\n" \
    "${WORKDIR}/logs/share_cf_rb_${SLURM_JOB_ID}_${SHARD_NAME}.out" \
    "${WORKDIR}/logs/share_cf_rb_${SLURM_JOB_ID}_${SHARD_NAME}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
