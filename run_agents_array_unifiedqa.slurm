#!/bin/bash -l
# MSI fair-share oriented job array for running the LongMemEval agents with the
# unified factual QA head. Slow agents are split into two 25-question shards so
# baseline completion time drops substantially without changing the underlying
# evaluation. Default concurrency is capped at 6 to balance time-to-results and
# API/rate-limit pressure.
#
# Example:
#   sbatch /users/9/chen7751/csci8980/run_agents_array_unifiedqa.slurm
#
# Monitor:
#   squeue -j <jobid>
#   sacct -j <jobid> --format=JobID,JobName,State,Elapsed,Start,End
#
# Array index mapping (ordered by runtime, longest first):
#   0=theanine_p1, 1=theanine_p2
#   2=share_p1,    3=share_p2
#   4=memoryos_p1, 5=memoryos_p2
#   6=ldagent_p1,  7=ldagent_p2
#   8=anna_full
#
# API key resolution:
#   Preferred: agent-specific vars (OPENAI_API_KEY_THEANINE, ...).
#   Practical fallback for %6 concurrency:
#     - THEANINE p1 -> OPENAI_API_KEY_1
#     - THEANINE p2 -> OPENAI_API_KEY_2
#     - SHARE p1    -> OPENAI_API_KEY_3
#     - SHARE p2    -> OPENAI_API_KEY_4
#     - MEMORYOS p1 -> OPENAI_API_KEY_5
#     - MEMORYOS p2 -> OPENAI_API_KEY
#     - LDAGENT p1  -> OPENAI_API_KEY_5
#     - LDAGENT p2  -> OPENAI_API_KEY_4
#     - ANNA        -> OPENAI_API_KEY_3
#   Final fallback is OPENAI_API_KEY.

#SBATCH --job-name=lme_uqa
#SBATCH --partition=msismall
#SBATCH --constraint=genoa
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12g
#SBATCH --tmp=10g
#SBATCH --array=0-8%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"
mkdir -p logs

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export ANONYMIZED_TELEMETRY=False
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
DATE_SUFFIX="$(date +%m_%d)"
RUN_TAG="${RUN_TAG:-s_50_unifiedqa_${DATE_SUFFIX}}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
PART_SIZE="${PART_SIZE:-25}"

ANNA_BRIDGE="${WORKDIR}/anna_longmemeval_bridge/run_infer.py"
SHARE_BRIDGE="${WORKDIR}/share_longmemeval_bridge/run_infer.py"
MEMOS_BRIDGE="${WORKDIR}/memos_longmemeval_bridge/run_infer.py"
LD_BRIDGE="${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py"
THEANINE_BRIDGE="${WORKDIR}/theanine_longmemeval_bridge/run_infer.py"

ANNA_REPO="${WORKDIR}/AnnaAgent"
SHARE_REPO="${WORKDIR}/SHARE"
MEMOS_REPO="${WORKDIR}/MemoryOS"
LD_REPO="${WORKDIR}/LD-Agent"
THEANINE_REPO="${WORKDIR}/Theanine"

TASK_SPECS=(
  "theanine|p1|0|25"
  "theanine|p2|25|25"
  "share|p1|0|25"
  "share|p2|25|25"
  "memoryos|p1|0|25"
  "memoryos|p2|25|25"
  "ldagent|p1|0|25"
  "ldagent|p2|25|25"
  "anna|full|0|0"
)

IFS='|' read -r AGENT PART_TAG OFFSET LIMIT <<< "${TASK_SPECS[$SLURM_ARRAY_TASK_ID]}"

sleep $((SLURM_ARRAY_TASK_ID * 45))

set_agent_key() {
  local specific_var="$1"
  local pool_var="${2:-}"
  local fallback="${OPENAI_API_KEY:-}"
  local specific="${!specific_var:-}"
  local pooled=""
  if [[ -n "${pool_var}" ]]; then
    pooled="${!pool_var:-}"
  fi
  if [[ -n "${specific}" ]]; then
    export OPENAI_API_KEY="${specific}"
  elif [[ -n "${pooled}" ]]; then
    export OPENAI_API_KEY="${pooled}"
  elif [[ -n "${fallback}" ]]; then
    export OPENAI_API_KEY="${fallback}"
  else
    echo "Missing OpenAI key: checked ${specific_var}${pool_var:+, ${pool_var}}, then OPENAI_API_KEY." >&2
    exit 1
  fi
}

echo "============================================================"
echo "[$(date)] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} agent=${AGENT} part=${PART_TAG} offset=${OFFSET} limit=${LIMIT}"
echo "host=$(hostname)"
echo "cwd=${WORKDIR}"
echo "============================================================"

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

OUTPUT_SUFFIX="${RUN_TAG}_${PART_TAG}"

case "${AGENT}" in
  anna)
    set_agent_key "OPENAI_API_KEY_ANNA" "OPENAI_API_KEY_3"
    conda activate anna-lme
    python "${ANNA_BRIDGE}" \
      --anna-agent-dir "${ANNA_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_anna_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --disable-full-tertiary-init \
      --disable-need-check \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}"
    ;;

  share)
    if [[ "${PART_TAG}" == "p1" ]]; then
      set_agent_key "OPENAI_API_KEY_SHARE" "OPENAI_API_KEY_3"
    else
      set_agent_key "OPENAI_API_KEY_SHARE" "OPENAI_API_KEY_4"
    fi
    conda activate share-lme
    python "${SHARE_BRIDGE}" \
      --share-dir "${SHARE_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_share_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --strict-selection-mode qa \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}"
    ;;

  memoryos)
    if [[ "${PART_TAG}" == "p1" ]]; then
      set_agent_key "OPENAI_API_KEY_MEMORYOS" "OPENAI_API_KEY_5"
    else
      set_agent_key "OPENAI_API_KEY_MEMORYOS" "OPENAI_API_KEY"
    fi
    conda activate memos-lme
    python "${MEMOS_BRIDGE}" \
      --memoryos-dir "${MEMOS_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_memoryos_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --reset-mode reinit \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}"
    ;;

  ldagent)
    if [[ "${PART_TAG}" == "p1" ]]; then
      set_agent_key "OPENAI_API_KEY_LDAGENT" "OPENAI_API_KEY_5"
    else
      set_agent_key "OPENAI_API_KEY_LDAGENT" "OPENAI_API_KEY_4"
    fi
    conda activate ld-lme
    python "${LD_BRIDGE}" \
      --ld-agent-dir "${LD_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_ldagent_${OUTPUT_SUFFIX}.trace.jsonl" \
      --openai-base-url "${OPENAI_BASE_URL}" \
      --llm-model "${LLM_MODEL}" \
      --session-gap-seconds 600 \
      --dist-thres 0.5527 \
      --no-force-flush-before-answer \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}"
    ;;

  theanine)
    if [[ "${PART_TAG}" == "p1" ]]; then
      set_agent_key "OPENAI_API_KEY_THEANINE" "OPENAI_API_KEY_1"
    else
      set_agent_key "OPENAI_API_KEY_THEANINE" "OPENAI_API_KEY_2"
    fi
    conda activate theanine-lme
    python "${THEANINE_BRIDGE}" \
      --theanine-dir "${THEANINE_REPO}" \
      --longmemeval-file "${DATA_FILE}" \
      --out-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.jsonl" \
      --trace-jsonl "${WORKDIR}/LongMemEval/preds_theanine_${OUTPUT_SUFFIX}.trace.jsonl" \
      --llm-model "${LLM_MODEL}" \
      "${LIMIT_ARGS[@]}" \
      "${OFFSET_ARGS[@]}"
    ;;

  *)
    echo "Unknown agent '${AGENT}'" >&2
    exit 1
    ;;
esac

echo "[$(date)] Completed agent=${AGENT}"
