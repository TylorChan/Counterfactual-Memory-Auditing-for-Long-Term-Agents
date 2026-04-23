#!/bin/bash -l
# THEANINE-only unified-QA baseline+CF run over all 50 questions, split into 10 parallel 5-question shards.

#SBATCH --job-name=the_uqacf
#SBATCH --partition=msismall
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=80g
#SBATCH --tmp=20g
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
RUN_TAG="${RUN_TAG:-s_50_unifiedqa_cf_theanine_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-30}"
JOBS="${JOBS:-${SLURM_NTASKS}}"
ENABLE_CF_WRAPPER="${ENABLE_CF_WRAPPER:-1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-2}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
COMMANDS_FILE="${LOG_ROOT}/the_uqacf_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${LOG_ROOT}/the_uqacf_${SLURM_JOB_ID}.parallel.log"
THEANINE_REPO_ROOT="${WORKDIR}/cf_runtime/${RUN_TAG}/theanine_repos"

mkdir -p "${THEANINE_REPO_ROOT}"
for shard in s1 s2 s3 s4 s5 s6 s7 s8 s9 s10; do
  repo="${THEANINE_REPO_ROOT}/Theanine_${shard}_repo"
  if [[ ! -d "${repo}" ]]; then
    cp -a "${WORKDIR}/Theanine" "${repo}"
  fi
done
export THEANINE_REPO_ROOT

TASK_SPECS=(
  "s1|0|5|OPENAI_API_KEY_1"
  "s2|5|5|OPENAI_API_KEY_2"
  "s3|10|5|OPENAI_API_KEY_3"
  "s4|15|5|OPENAI_API_KEY_4"
  "s5|20|5|OPENAI_API_KEY_5"
  "s6|25|5|OPENAI_API_KEY_6"
  "s7|30|5|OPENAI_API_KEY_7"
  "s8|35|5|OPENAI_API_KEY_8"
  "s9|40|5|OPENAI_API_KEY_9"
  "s10|45|5|OPENAI_API_KEY_10"
)

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} launching THEANINE unified-QA baseline+CF 10x5"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "run_tag=${RUN_TAG}"
echo "theanine_repo_root=${THEANINE_REPO_ROOT}"
echo "parallel_jobs=${JOBS}"
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
    "theanine" \
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
    "${LOG_ROOT}/the_uqacf_${SLURM_JOB_ID}_${PART_TAG}.out" \
    "${LOG_ROOT}/the_uqacf_${SLURM_JOB_ID}_${PART_TAG}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
