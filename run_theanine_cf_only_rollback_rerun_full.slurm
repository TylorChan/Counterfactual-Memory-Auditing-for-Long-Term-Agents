#!/bin/bash -l
# Full rerun of THEANINE CF-only rollback over the 48 baseline-available questions.

#SBATCH --job-name=the_cf_rb_rerun
#SBATCH --partition=msismall
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80g
#SBATCH --tmp=20g
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
CF_TAG="${CF_TAG:-the_cf_rb2_rerun_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-30}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-2}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"
COMMANDS_FILE="${WORKDIR}/logs/the_cf_rb_rerun_${SLURM_JOB_ID}.commands.txt"
PARALLEL_JOBLOG="${WORKDIR}/logs/the_cf_rb_rerun_${SLURM_JOB_ID}.parallel.log"
THEANINE_REPO_ROOT="${WORKDIR}/cf_only_runtime/${CF_TAG}/theanine_repos"

mkdir -p "${THEANINE_REPO_ROOT}"
for shard in s1 s2 s3 s4 s5; do
  repo="${THEANINE_REPO_ROOT}/Theanine_${shard}_repo"
  if [[ ! -d "${repo}" ]]; then
    cp -a "${WORKDIR}/Theanine" "${repo}"
  fi
done

TASK_SPECS=(
  "s1|0|10|OPENAI_API_KEY_1"
  "s2|10|10|OPENAI_API_KEY_2"
  "s3|20|10|OPENAI_API_KEY_3"
  "s4|30|10|OPENAI_API_KEY_4"
  "s5|40|8|OPENAI_API_KEY_5"
)

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} rerunning THEANINE rollback-only CF 48q"
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
  printf "%q " bash "${WORKDIR}/scripts/run_theanine_cf_only_5x10_shard.sh" \
    "${WORKDIR}" \
    "${BASELINE_TAG}" \
    "${CF_TAG}" \
    "${THEANINE_REPO_ROOT}" \
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
    "${WORKDIR}/logs/the_cf_rb_rerun_${SLURM_JOB_ID}_${SHARD_NAME}.out" \
    "${WORKDIR}/logs/the_cf_rb_rerun_${SLURM_JOB_ID}_${SHARD_NAME}.err" >> "${COMMANDS_FILE}"
  i=$((i + 1))
done

parallel --jobs "${JOBS}" --joblog "${PARALLEL_JOBLOG}" < "${COMMANDS_FILE}"
