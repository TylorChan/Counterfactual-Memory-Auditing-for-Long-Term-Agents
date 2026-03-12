#!/bin/bash -l
#SBATCH --job-name=lme_ther
#SBATCH --partition=msismall
#SBATCH --constraint=genoa
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=24g
#SBATCH --tmp=20g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
## #SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"

DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
THEANINE_BRIDGE="${WORKDIR}/theanine_longmemeval_bridge/run_infer.py"
THEANINE_REPO="${WORKDIR}/Theanine"

# Resume from question 48 (0-based offset 47) for the remaining 3 questions.
LIMIT="${LIMIT:-3}"
OFFSET="${OFFSET:-47}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
RUN_TAG="${RUN_TAG:-s_50_fullhist_resume47}"
HISTORY_SESSIONS="${HISTORY_SESSIONS:-0}"
PRESERVE_SESSION_ORDER="${PRESERVE_SESSION_ORDER:-0}"
OMIT_QUESTION_DATE="${OMIT_QUESTION_DATE:-0}"
VERBOSE_UPSTREAM="${VERBOSE_UPSTREAM:-0}"

THEANINE_OUT="${WORKDIR}/LongMemEval/preds_theanine_${RUN_TAG}.jsonl"
THEANINE_TRACE="${WORKDIR}/LongMemEval/preds_theanine_${RUN_TAG}.trace.jsonl"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export ANONYMIZED_TELEMETRY=False

if [[ "${PRESERVE_SESSION_ORDER}" == "1" ]]; then
  ORDER_ARGS=(--preserve-session-order)
else
  ORDER_ARGS=()
fi

if [[ "${OMIT_QUESTION_DATE}" == "1" ]]; then
  QUESTION_DATE_ARGS=(--omit-question-date)
else
  QUESTION_DATE_ARGS=()
fi

if [[ "${VERBOSE_UPSTREAM}" == "1" ]]; then
  VERBOSE_ARGS=(--verbose-upstream)
else
  VERBOSE_ARGS=()
fi

echo "============================================================"
echo "================== AGENT: THEANINE RESUME =================="
echo "============================================================"
echo "[$(date)] Resume THEANINE from offset ${OFFSET} limit ${LIMIT}"

conda activate theanine-lme
python "${THEANINE_BRIDGE}" \
  --theanine-dir "${THEANINE_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${THEANINE_OUT}" \
  --trace-jsonl "${THEANINE_TRACE}" \
  --llm-model "${LLM_MODEL}" \
  --history-sessions "${HISTORY_SESSIONS}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --fail-fast \
  "${ORDER_ARGS[@]}" \
  "${QUESTION_DATE_ARGS[@]}" \
  "${VERBOSE_ARGS[@]}"

echo "[$(date)] Done"
echo "THEANINE resume out  : ${THEANINE_OUT}"
echo "THEANINE resume trace: ${THEANINE_TRACE}"
