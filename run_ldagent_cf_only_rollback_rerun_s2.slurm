#!/bin/bash -l
# Rerun only the failed LD-Agent CF-only rollback shard s2 (questions 10-19).

#SBATCH --job-name=ld_cf_rb_s2
#SBATCH --partition=msismall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --tmp=6g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"
mkdir -p logs

DATE_SUFFIX="$(date +%m_%d_%H_%M)"
BASELINE_TAG="${BASELINE_TAG:-03_14_22_46}"
CF_TAG="${CF_TAG:-ld_cf_rb3_rerun_s2_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
CF_TARGET_SCOPE="${CF_TARGET_SCOPE:-prompt}"
CF_MAX_WRITES="${CF_MAX_WRITES:-3}"
CF_DOMINANCE_THRESHOLD="${CF_DOMINANCE_THRESHOLD:-0.75}"

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} rerunning LD-Agent rollback-only CF shard s2"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "baseline_tag=${BASELINE_TAG}"
echo "cf_tag=${CF_TAG}"
echo "cf_target_scope=${CF_TARGET_SCOPE}"
echo "cf_max_writes=${CF_MAX_WRITES}"
echo "cf_dominance_threshold=${CF_DOMINANCE_THRESHOLD}"
echo "============================================================"

bash "${WORKDIR}/scripts/run_ldagent_cf_only_5x10_shard.sh" \
  "${WORKDIR}" \
  "${BASELINE_TAG}" \
  "${CF_TAG}" \
  "s2" \
  "10" \
  "10" \
  "OPENAI_API_KEY_2" \
  "${LLM_MODEL}" \
  "${OPENAI_BASE_URL}" \
  "0" \
  "${CF_TARGET_SCOPE}" \
  "${CF_MAX_WRITES}" \
  "${CF_DOMINANCE_THRESHOLD}"
