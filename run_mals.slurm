#!/bin/bash -l
#SBATCH --job-name=lme_mals
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
## Optional (set your own x500 mail):
## #SBATCH --mail-user=chen7751@umn.edu

set -euo pipefail

# MSI best practice: move to your project working directory first.
WORKDIR="/users/9/chen7751/csci8980"
cd "${WORKDIR}"

# Shared experiment config
DATA_FILE="${WORKDIR}/LongMemEval/data/longmemeval_s_cleaned_50.json"
ANNA_BRIDGE="${WORKDIR}/anna_longmemeval_bridge/run_infer.py"
LD_BRIDGE="${WORKDIR}/ldagent_longmemeval_bridge/run_infer.py"
SHARE_BRIDGE="${WORKDIR}/share_longmemeval_bridge/run_infer.py"
MEMOS_BRIDGE="${WORKDIR}/memos_longmemeval_bridge/run_infer.py"
ANNA_REPO="${WORKDIR}/AnnaAgent"
LD_REPO="${WORKDIR}/LD-Agent"
SHARE_REPO="${WORKDIR}/SHARE"
MEMOS_REPO="${WORKDIR}/MemoryOS"

# You can override these when submitting, e.g.:
# sbatch --export=ALL,LIMIT=1,LLM_MODEL=gpt-4o-mini run_anna_then_share.slurm
LIMIT="${LIMIT:-0}"                # 0 => full 50 questions
OFFSET="${OFFSET:-0}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
RUN_TAG="${RUN_TAG:-s_50}"
SHARE_STRICT_SELECTION_MODE="${SHARE_STRICT_SELECTION_MODE:-qa}"

ANNA_OUT="${WORKDIR}/LongMemEval/preds_anna_${RUN_TAG}.jsonl"
ANNA_TRACE="${WORKDIR}/LongMemEval/preds_anna_${RUN_TAG}.trace.jsonl"
LD_OUT="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.jsonl"
LD_TRACE="${WORKDIR}/LongMemEval/preds_ldagent_${RUN_TAG}.trace.jsonl"
SHARE_OUT="${WORKDIR}/LongMemEval/preds_share_${RUN_TAG}.jsonl"
SHARE_TRACE="${WORKDIR}/LongMemEval/preds_share_${RUN_TAG}.trace.jsonl"
MEMOS_OUT="${WORKDIR}/LongMemEval/preds_memoryos_${RUN_TAG}.jsonl"
MEMOS_TRACE="${WORKDIR}/LongMemEval/preds_memoryos_${RUN_TAG}.trace.jsonl"

# Conda init (adjust if your conda install path is different on MSI)
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

# Disable Chroma anonymized telemetry to avoid posthog capture warnings.
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

echo "============================================================"
echo "======================== AGENT: ANNA ========================"
echo "============================================================"
echo "[$(date)] Run Anna"
conda activate anna-lme
python "${ANNA_BRIDGE}" \
  --anna-agent-dir "${ANNA_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${ANNA_OUT}" \
  --trace-jsonl "${ANNA_TRACE}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --disable-full-tertiary-init \
  --disable-need-check \
  --fail-fast \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}"

echo "============================================================"
echo "======================== AGENT: SHARE ======================="
echo "============================================================"
echo "[$(date)] Run SHARE"
conda activate share-lme
python "${SHARE_BRIDGE}" \
  --share-dir "${SHARE_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${SHARE_OUT}" \
  --trace-jsonl "${SHARE_TRACE}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --strict-selection-mode "${SHARE_STRICT_SELECTION_MODE}" \
  --fail-fast \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}"

echo "============================================================"
echo "====================== AGENT: MEMORYOS ======================"
echo "============================================================"
echo "[$(date)] Run MemoryOS"
conda activate memos-lme
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "${MEMOS_BRIDGE}" \
  --memoryos-dir "${MEMOS_REPO}" \
  --longmemeval-file "${DATA_FILE}" \
  --out-jsonl "${MEMOS_OUT}" \
  --trace-jsonl "${MEMOS_TRACE}" \
  --openai-base-url "${OPENAI_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --reset-mode reinit \
  --fail-fast \
  "${LIMIT_ARGS[@]}" \
  "${OFFSET_ARGS[@]}"

echo "============================================================"
echo "====================== AGENT: LD-AGENT ======================"
echo "============================================================"
echo "[$(date)] Run LD-Agent"
conda activate ld-lme
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

echo "[$(date)] Done"
echo "Anna output : ${ANNA_OUT}"
echo "Share output: ${SHARE_OUT}"
echo "MemoryOS out: ${MEMOS_OUT}"
echo "LD output   : ${LD_OUT}"
