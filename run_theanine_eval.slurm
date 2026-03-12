#!/bin/bash -l
#SBATCH --job-name=lme_thev
#SBATCH --partition=msismall
#SBATCH --constraint=genoa
#SBATCH --time=96:00:00
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
EVAL_SCRIPT="${WORKDIR}/LongMemEval/src/evaluation/evaluate_qa.py"

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"
RUN_TAG="${RUN_TAG:-s_50_fullhist}"
RUN_EVAL="${RUN_EVAL:-1}"
HISTORY_SESSIONS="${HISTORY_SESSIONS:-0}"
PRESERVE_SESSION_ORDER="${PRESERVE_SESSION_ORDER:-0}"
OMIT_QUESTION_DATE="${OMIT_QUESTION_DATE:-0}"
VERBOSE_UPSTREAM="${VERBOSE_UPSTREAM:-0}"

THEANINE_OUT="${WORKDIR}/LongMemEval/preds_theanine_${RUN_TAG}.jsonl"
THEANINE_TRACE="${WORKDIR}/LongMemEval/preds_theanine_${RUN_TAG}.trace.jsonl"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export ANONYMIZED_TELEMETRY=False

if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
else
  LIMIT_ARGS=()
fi

if [[ "${OFFSET}" != "0" ]]; then
  OFFSET_ARGS=(--offset "${OFFSET}")
else
  OFFSET_ARGS=()
fi

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
echo "====================== AGENT: THEANINE ======================"
echo "============================================================"
echo "[$(date)] Run THEANINE"

conda activate theanine-lme
python "${THEANINE_BRIDGE}" \
  --theanine-dir "${THEANINE_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${THEANINE_OUT}" \
  --trace-jsonl "${THEANINE_TRACE}" \
  --llm-model "${LLM_MODEL}" \
  --history-sessions "${HISTORY_SESSIONS}" \
  --fail-fast \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}" \
  "${ORDER_ARGS[@]}" \
  "${QUESTION_DATE_ARGS[@]}" \
  "${VERBOSE_ARGS[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "============================================================"
  echo "====================== EVAL: THEANINE ======================="
  echo "============================================================"
  echo "[$(date)] Evaluate THEANINE"
  conda activate longmemeval-lite
  python "${EVAL_SCRIPT}" "${EVAL_MODEL}" "${THEANINE_OUT}" "${DATA_FILE}"
fi

echo "[$(date)] Done"
echo "THEANINE out  : ${THEANINE_OUT}"
echo "THEANINE trace: ${THEANINE_TRACE}"
