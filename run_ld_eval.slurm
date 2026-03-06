#!/bin/bash -l
#SBATCH --job-name=lme_ldev
#SBATCH --partition=msismall
#SBATCH --time=24:00:00
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
LD_BRIDGE="${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py"
LD_REPO="${WORKDIR}/LD-Agent"
EVAL_SCRIPT="${WORKDIR}/LongMemEval/src/evaluation/evaluate_qa.py"

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
RUN_TAG="${RUN_TAG:-s_50}"
RUN_EVAL="${RUN_EVAL:-1}"

LD_OUT="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.jsonl"
LD_TRACE="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.trace.jsonl"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ld-lme

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

echo "========== LD-AGENT INFER =========="
python "${LD_BRIDGE}" \
  --ld-agent-dir "${LD_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${LD_OUT}" \
  --trace-jsonl "${LD_TRACE}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --session-gap-seconds 600 \
  --dist-thres 0.5527 \
  --no-force-flush-before-answer \
  --fail-fast \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "========== LD-AGENT EVAL =========="
  conda activate longmemeval-lite
  python "${EVAL_SCRIPT}" "${EVAL_MODEL}" "${LD_OUT}" "${DATA_FILE}"
fi

echo "Done: ${LD_OUT}"
